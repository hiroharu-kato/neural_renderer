import string

import chainer
import chainer.functions as cf


class RasterizeSilhouette(chainer.Function):
    def __init__(self, image_size, near, far, eps):
        super(RasterizeSilhouette, self).__init__()
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps

        self.face_index_map = None
        self.images = None

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)
        faces_type, = in_types
        chainer.utils.type_check.expect(
            faces_type.dtype.kind == 'f',
            faces_type.ndim == 4,
            faces_type.shape[2] == 3,
            faces_type.shape[3] == 3,
        )

    def forward_gpu(self, inputs):
        xp = chainer.cuda.get_array_module(inputs[0])
        faces = inputs[0]
        faces = xp.ascontiguousarray(faces)
        batch_size, num_faces = faces.shape[:2]
        image_size = self.image_size

        # initialize buffers
        self.face_index_map = -1 * xp.ascontiguousarray(xp.ones((batch_size, image_size, image_size), dtype='int32'))
        self.images = xp.ascontiguousarray(xp.zeros((batch_size, image_size, image_size), dtype='float32'))

        # vertices -> face_index_map, z_map
        # face_index_map = -1 if background
        loop = xp.arange(batch_size * num_faces).astype('int32')
        lock = xp.zeros(self.images.shape, 'int32')
        z_map = xp.zeros(self.images.shape, 'float32') + self.far
        chainer.cuda.elementwise(
            'int32 j, raw float32 faces, raw int32 face_index_map, raw float32 images, raw float32 z_map, ' +
            'raw int32 lock',
            '',
            string.Template('''
                const int bn = i / ${num_faces};
                const int fn = i % ${num_faces};
                const int is = ${image_size};
                const float* face = &faces[i * 9];

                /* check back */
                if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0])) return;

                /* pi[0], pi[1], pi[2] = leftmost, middle, rightmost points */
                int pi[3];
                if (face[0] < face[3]) {
                    if (face[6] < face[0]) pi[0] = 2; else pi[0] = 0;
                    if (face[3] < face[6]) pi[2] = 2; else pi[2] = 1;
                } else {
                    if (face[6] < face[3]) pi[0] = 2; else pi[0] = 1;
                    if (face[0] < face[6]) pi[2] = 2; else pi[2] = 0;
                }
                for (int k = 0; k < 3; k++) if (pi[0] != k && pi[2] != k) pi[1] = k;

                /* p[num][xyz]: x, y is normalized from [-1, 1] to [0, is - 1]. */
                float p[3][3];
                for (int num = 0; num < 3; num++) {
                    for (int dim = 0; dim < 3; dim++) {
                        if (dim != 2) {
                            p[num][dim] = 0.5 * (face[3 * pi[num] + dim] * is + is - 1);
                        } else {
                            p[num][dim] = face[3 * pi[num] + dim];
                        }
                    }
                }
                if (p[0][0] == p[2][0]) return; // line, not triangle

                /* compute face_inv */
                float face_inv[9] = {
                    p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
                    p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
                    p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
                float face_inv_denominator = (
                    p[2][0] * (p[0][1] - p[1][1]) +
                    p[0][0] * (p[1][1] - p[2][1]) +
                    p[1][0] * (p[2][1] - p[0][1]));
                for (int k = 0; k < 9; k++) face_inv[k] /= face_inv_denominator;

                /* from left to right */
                const int xi_min = max(ceil(p[0][0]), 0.);
                const int xi_max = min(p[2][0], is - 1.);
                for (int xi = xi_min; xi <= xi_max; xi++) {
                    /* compute yi_min and yi_max */
                    float yi1, yi2;
                    if (xi <= p[1][0]) {
                        if (p[1][0] - p[0][0] != 0) {
                            yi1 = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];
                        } else {
                            yi1 = p[1][1];
                        }
                    } else {
                        if (p[2][0] - p[1][0] != 0) {
                            yi1 = (p[2][1] - p[1][1]) / (p[2][0] - p[1][0]) * (xi - p[1][0]) + p[1][1];
                        } else {
                            yi1 = p[1][1];
                        }
                    }
                    yi2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];

                    /* from up to bottom */
                    int yi_min = max(0., ceil(min(yi1, yi2)));
                    int yi_max = min(max(yi1, yi2), is - 1.);
                    for (int yi = yi_min; yi <= yi_max; yi++) {
                        int index = bn * is * is + yi * is + xi;

                        /* compute w = face_inv * p */
                        float w[3];
                        for (int k = 0; k < 3; k++) w[k] = (
                            face_inv[3 * k + 0] * xi +
                            face_inv[3 * k + 1] * yi +
                            face_inv[3 * k + 2]);

                        /* sum(w) -> 1, 0 < w < 1 */
                        float w_sum = 0;
                        for (int k = 0; k < 3; k++) {
                            if (w[k] < 0) w[k] = 0;
                            if (1 < w[k]) w[k] = 1;
                            w_sum += w[k];
                        }
                        for (int k = 0; k < 3; k++) w[k] /= w_sum;

                        /* compute 1 / zp = sum(w / z) */
                        const float zp = 1. / (w[0] / p[0][2] + w[1] / p[1][2] + w[2] / p[2][2]);
                        if (zp <= ${near} || ${far} <= zp) continue;

                        /* lock and update */
                        bool locked = false;
                        do {
                            if (locked = atomicCAS(&lock[index], 0, 1) == 0) {
                                if (zp < z_map[index]) {
                                    atomicExch(&images[index], 1);
                                    atomicExch(&face_index_map[index], fn);
                                    atomicExch(&z_map[index], zp);
                                }
                            }
                            if (locked) atomicExch(&lock[index], 0);
                        } while (!locked);
                    }
                }
            ''').substitute(
                num_faces=num_faces,
                image_size=image_size,
                near=self.near,
                far=self.far,
            ),
            'function',
        )(loop, faces.ravel(), self.face_index_map.ravel(), self.images.ravel(), z_map, lock)

        return self.images,

    def backward_gpu(self, inputs, grad_outputs):
        xp = chainer.cuda.get_array_module(inputs[0])
        faces = xp.ascontiguousarray(inputs[0])
        grad_images = xp.ascontiguousarray(grad_outputs[0])
        grad_faces = xp.ascontiguousarray(xp.zeros_like(faces, dtype='float32'))
        batch_size, num_faces = faces.shape[:2]
        image_size = self.image_size

        # pseudo gradient
        loop = xp.arange(batch_size * num_faces).astype('int32')
        chainer.cuda.elementwise(
            'int32 j, raw float32 faces, raw int32 face_index_map, raw float32 images, raw float32 grad_images, ' +
            'raw float32 grad_faces',
            '',
            string.Template('''
                const int bn = i / ${num_faces};
                const int fn = i % ${num_faces};
                const int is = ${image_size};
                const float* face = &faces[i * 9];
                float grad_face[9] = {};
                
                /* check backside */
                if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0])) return;

                /* for each edge */
                for (int edge_num = 0; edge_num < 3; edge_num++) {
                    /* set points of target edge */
                    const int p0_i = edge_num % 3;
                    const int p1_i = (edge_num + 1) % 3;
                    const int p2_i = (edge_num + 2) % 3;
                    const float p0_xi = 0.5 * (face[3 * p0_i + 0] * is + is - 1);
                    const float p0_yi = 0.5 * (face[3 * p0_i + 1] * is + is - 1);
                    const float p1_xi = 0.5 * (face[3 * p1_i + 0] * is + is - 1);
                    const float p1_yi = 0.5 * (face[3 * p1_i + 1] * is + is - 1);
                    const float p2_xi = 0.5 * (face[3 * p2_i + 0] * is + is - 1);
                    const float p2_yi = 0.5 * (face[3 * p2_i + 1] * is + is - 1);
                    /* for dy, dx */
                    for (int axis = 0; axis < 2; axis++) {
                        /* */
                        float p0_d0, p0_d1, p1_d0, p1_d1, p2_d0, p2_d1;
                        if (axis == 0) {
                            p0_d0 = p0_xi;
                            p0_d1 = p0_yi;
                            p1_d0 = p1_xi;
                            p1_d1 = p1_yi;
                            p2_d0 = p2_xi;
                            p2_d1 = p2_yi;
                        } else {
                            p0_d0 = p0_yi;
                            p0_d1 = p0_xi;
                            p1_d0 = p1_yi;
                            p1_d1 = p1_xi;
                            p2_d0 = p2_yi;
                            p2_d1 = p2_xi;
                        }
                        /* set direction */
                        int direction;
                        if (axis == 0) {
                            if (p0_d0 < p1_d0) direction = -1; else direction = 1;
                        } else {
                            if (p0_d0 < p1_d0) direction = 1; else direction = -1;
                        }
                        /* along edge */
                        int d0_from, d0_to;
                        d0_from = max(ceil(min(p0_d0, p1_d0)), 0.);
                        d0_to = min(max(p0_d0, p1_d0), is - 1.);
                        for (int d0 = d0_from; d0 <= d0_to; d0++) {
                            /* get cross point */
                            int d1_in, d1_out;
                            const float d1_cross = (p1_d1 - p0_d1) / (p1_d0 - p0_d0) * (d0 - p0_d0) + p0_d1;
                            if (0 < direction) d1_in = floor(d1_cross); else d1_in = ceil(d1_cross);
                            d1_out = d1_in + direction;
                            /* continue if cross point is not shown */
                            if (d1_in < 0 || is <= d1_in) continue;
                            if (d1_out < 0 || is <= d1_out) continue;
                            float color_in, color_out;
                            if (axis == 0) {
                                color_in = images[bn * is * is + d1_in * is + d0];
                                color_out = images[bn * is * is + d1_out * is + d0];
                            } else {
                                color_in = images[bn * is * is + d0 * is + d1_in];
                                color_out = images[bn * is * is + d0 * is + d1_out];
                            }
                            /* out */
                            bool is_in_fn = true;
                            if (axis == 0) {
                                if (face_index_map[bn * is * is + d1_in * is + d0] != fn) is_in_fn = false;
                            } else {
                                if (face_index_map[bn * is * is + d0 * is + d1_in] != fn) is_in_fn = false;
                            }
                            if (is_in_fn) {
                                int d1_limit;
                                if (0 < direction) d1_limit = is - 1; else d1_limit = 0;
                                int d1_from = max(min(d1_out, d1_limit), 0);
                                int d1_to = min(max(d1_out, d1_limit), is - 1);
                                float* images_p;
                                float* grad_images_p;
                                if (axis == 0) {
                                    images_p = &images[bn * is * is + d1_from * is + d0] - is;
                                    grad_images_p = &grad_images[bn * is * is + d1_from * is + d0] - is;
                                } else {
                                    images_p = &images[bn * is * is + d0 * is + d1_from] - 1;
                                    grad_images_p = &grad_images[bn * is * is + d0 * is + d1_from] - 1;
                                }
                                for (int d1 = d1_from; d1 <= d1_to; d1++) {
                                    float diff, grad;
                                    if (axis == 0) {
                                        images_p += is;
                                        grad_images_p += is;
                                        diff = *images_p - color_in;
                                        grad = *grad_images_p;
                                    } else {
                                        images_p += 1;
                                        grad_images_p += 1;
                                        diff = *images_p - color_in;
                                        grad = *grad_images_p;
                                    }
                                    if (diff * grad <= 0) continue;
                                    if (p1_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (p1_d0 - d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p0_i * 3 + (1 - axis)] -= grad * diff / dist;
                                    }
                                    if (p0_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (d0 - p0_d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p1_i * 3 + (1 - axis)] -= grad * diff / dist;
                                    }
                                }
                            }
                            /* in */
                            {
                                int d1_limit;
                                float d0_cross2;
                                if ((d0 - p0_d0) * (d0 - p2_d0) < 0) {
                                    d0_cross2 = (p2_d1 - p0_d1) / (p2_d0 - p0_d0) * (d0 - p0_d0) + p0_d1;
                                } else {
                                    d0_cross2 = (p1_d1 - p2_d1) / (p1_d0 - p2_d0) * (d0 - p2_d0) + p2_d1;
                                }
                                if (0 < direction) d1_limit = ceil(d0_cross2); else d1_limit = floor(d0_cross2);
                                int d1_from = max(min(d1_in, d1_limit), 0);
                                int d1_to = min(max(d1_in, d1_limit), is - 1);
                                int* face_index_map_p;
                                float* images_p;
                                float* grad_images_p;
                                if (axis == 0) {
                                    face_index_map_p = &face_index_map[bn * is * is + d1_from * is + d0] - is;
                                    images_p = &images[bn * is * is + d1_from * is + d0] - is;
                                    grad_images_p = &grad_images[bn * is * is + d1_from * is + d0] - is;
                                } else {
                                    face_index_map_p = &face_index_map[bn * is * is + d0 * is + d1_from] - 1;
                                    images_p = &images[bn * is * is + d0 * is + d1_from] - 1;
                                    grad_images_p = &grad_images[bn * is * is + d0 * is + d1_from] - 1;
                                }
                                for (int d1 = d1_from; d1 <= d1_to; d1++) {
                                    float diff, grad;
                                    if (axis == 0) {
                                        face_index_map_p += is;
                                        images_p += is;
                                        grad_images_p += is;
                                        if (*face_index_map_p != fn) continue;
                                        diff = *images_p - color_out;
                                        grad = *grad_images_p;
                                    } else {
                                        face_index_map_p += 1;
                                        images_p += 1;
                                        grad_images_p += 1;
                                        if (*face_index_map_p != fn) continue;
                                        diff = *images_p - color_out;
                                        grad = *grad_images_p;
                                    }
                                    if (diff * grad <= 0) continue;
                                    if (p1_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (p1_d0 - d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p0_i * 3 + (1 - axis)] -= grad * diff / dist;
                                    }
                                    if (p0_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (d0 - p0_d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p1_i * 3 + (1 - axis)] -= grad * diff / dist;
                                    }
                                }
                            }
                        }
                    }
                }
                /* set to global gradient variable */
                for (int k = 0; k < 9; k++) grad_faces[i * 9 + k] = grad_face[k];
            ''').substitute(
                image_size=image_size,
                num_faces=num_faces,
                eps=self.eps,
            ),
            'function',
        )(loop, faces, self.face_index_map, self.images, grad_images.ravel(), grad_faces)

        return grad_faces,

    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError


def rasterize_silhouettes(faces, image_size=256, anti_aliasing=True, near=0.1, far=100, eps=1e-3):
    if anti_aliasing:
        images = RasterizeSilhouette(image_size * 2, near, far, eps)(faces)
        images = cf.average_pooling_2d(images[:, None, :, :], 2, 2)[:, 0]
    else:
        images = RasterizeSilhouette(image_size, near, far, eps)(faces)
    images = images[:, ::-1, :]
    return images
