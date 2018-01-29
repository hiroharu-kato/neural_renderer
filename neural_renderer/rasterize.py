import chainer
import chainer.functions as cf
import string


class Rasterize(chainer.Function):
    def __init__(self, image_size, near, far, eps, background_color):
        super(Rasterize, self).__init__()
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color

        self.face_index_map = None
        self.weight_map = None
        self.face_map = None
        self.z_map = None
        self.sampling_weight_map = None
        self.sampling_index_map = None
        self.images = None

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 2)
        faces_type, textures_type = in_types

        chainer.utils.type_check.expect(
            faces_type.dtype.kind == 'f',
            faces_type.ndim == 4,
            faces_type.shape[2] == 3,
            faces_type.shape[3] == 3,
        )

        chainer.utils.type_check.expect(
            textures_type.dtype.kind == 'f',
            textures_type.ndim == 6,
            textures_type.shape[2] == textures_type.shape[3],
            textures_type.shape[3] == textures_type.shape[4],
            textures_type.shape[5] == 3,
        )

        chainer.utils.type_check.expect(
            faces_type.shape[0] == textures_type.shape[0],
            faces_type.shape[1] == textures_type.shape[1],
        )

    def forward_gpu(self, inputs):
        xp = chainer.cuda.get_array_module(inputs[0])
        faces = xp.ascontiguousarray(inputs[0])
        textures = xp.ascontiguousarray(inputs[1])
        bs, nf = faces.shape[:2]
        is_ = self.image_size
        ts = textures.shape[2]

        # initialize buffers
        self.face_index_map = xp.ascontiguousarray(xp.zeros((bs, is_, is_), dtype='int32')) - 1
        self.weight_map = xp.ascontiguousarray(xp.zeros((bs, is_, is_, 3), dtype='float32'))
        self.face_map = xp.ascontiguousarray(xp.zeros((bs, is_, is_, 3, 3), dtype='float32'))
        self.z_map = xp.ascontiguousarray(xp.zeros((bs, is_, is_), dtype='float32')) + self.far
        self.sampling_weight_map = xp.ascontiguousarray(xp.zeros((bs, is_, is_, 8), 'float32'))
        self.sampling_index_map = xp.ascontiguousarray(xp.zeros((bs, is_, is_, 8), 'int32'))
        self.images = xp.ascontiguousarray(xp.zeros((bs, is_, is_, 3), dtype='float32'))

        # vertices -> face_index_map, z_map
        # face_index_map = -1 if background
        loop = xp.arange(bs * nf).astype('int32')
        lock = xp.zeros(self.images.shape, 'uint64')
        chainer.cuda.elementwise(
            'int32 j, raw float32 faces, raw int32 face_index_map, raw float32 weight_map, raw float32 face_map, ' +
            'raw float32 z_map, raw uint64 lock',
            '',
            string.Template('''
                const int bn = i / ${num_faces};
                const int fn = i % ${num_faces};
                const int is = ${image_size};
                const float* face = &faces[i * 9];

                /* check back */
                if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0])) return;

                /* p0, p1, p2 = leftmost, middle, rightmost points */
                int p0_i, p1_i, p2_i;
                if (face[0] < face[3]) {
                    if (face[6] < face[0]) p0_i = 2; else p0_i = 0;
                    if (face[3] < face[6]) p2_i = 2; else p2_i = 1;
                } else {
                    if (face[6] < face[3]) p0_i = 2; else p0_i = 1;
                    if (face[0] < face[6]) p2_i = 2; else p2_i = 0;
                }
                for (int k = 0; k < 3; k++) if (p0_i != k && p2_i != k) p1_i = k;
                const float p0_xi = 0.5 * (face[3 * p0_i + 0] * is + is - 1);
                const float p0_yi = 0.5 * (face[3 * p0_i + 1] * is + is - 1);
                const float p1_xi = 0.5 * (face[3 * p1_i + 0] * is + is - 1);
                const float p1_yi = 0.5 * (face[3 * p1_i + 1] * is + is - 1);
                const float p2_xi = 0.5 * (face[3 * p2_i + 0] * is + is - 1);
                const float p2_yi = 0.5 * (face[3 * p2_i + 1] * is + is - 1);
                const float p0_zp = face[3 * p0_i + 2];
                const float p1_zp = face[3 * p1_i + 2];
                const float p2_zp = face[3 * p2_i + 2];
                if (p0_xi == p2_xi) return; // line, not triangle

                /* compute face_inv */
                float face_inv[9] = {
                    p1_yi - p2_yi, p2_xi - p1_xi, p1_xi * p2_yi - p2_xi * p1_yi,
                    p2_yi - p0_yi, p0_xi - p2_xi, p2_xi * p0_yi - p0_xi * p2_yi,
                    p0_yi - p1_yi, p1_xi - p0_xi, p0_xi * p1_yi - p1_xi * p0_yi};
                float face_inv_denominator = (
                    p2_xi * (p0_yi - p1_yi) +
                    p0_xi * (p1_yi - p2_yi) +
                    p1_xi * (p2_yi - p0_yi));
                for (int k = 0; k < 9; k++) face_inv[k] /= face_inv_denominator;

                /* from left to right */
                const int xi_min = max(ceil(p0_xi), 0.);
                const int xi_max = min(p2_xi, is - 1.);
                for (int xi = xi_min; xi <= xi_max; xi++) {
                    /* compute yi_min and yi_max */
                    float yi1, yi2;
                    if (xi <= p1_xi) {
                        if (p1_xi - p0_xi != 0) {
                            yi1 = (p1_yi - p0_yi) / (p1_xi - p0_xi) * (xi - p0_xi) + p0_yi;
                        } else {
                            yi1 = p1_yi;
                        }
                    } else {
                        if (p2_xi - p1_xi != 0) {
                            yi1 = (p2_yi - p1_yi) / (p2_xi - p1_xi) * (xi - p1_xi) + p1_yi;
                        } else {
                            yi1 = p1_yi;
                        }
                    }
                    yi2 = (p2_yi - p0_yi) / (p2_xi - p0_xi) * (xi - p0_xi) + p0_yi;

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
                        const float zp = 1. / (w[0] / p0_zp + w[1] / p1_zp + w[2] / p2_zp);
                        if (zp <= ${near} || ${far} <= zp) continue;

                        /* lock and update */
                        bool locked = false;
                        do {
                            if (locked = atomicCAS(&lock[index], 0, 1) == 0) {
                                if (zp < z_map[index]) {
                                    face_index_map[index] = fn;
                                    z_map[index] = zp;
                                    weight_map[3 * index + p0_i] = w[0];
                                    weight_map[3 * index + p1_i] = w[1];
                                    weight_map[3 * index + p2_i] = w[2];
                                    for (int k = 0; k < 9; k++) face_map[9 * index + k] = face[k];
                                }
                            }
                            if (locked) atomicExch(&lock[index], 0);
                        } while (!locked);
                    }
                }
            ''').substitute(
                num_faces=nf,
                image_size=is_,
                near=self.near,
                far=self.far,
            ),
            'function',
        )(
            loop, faces.ravel(), self.face_index_map.ravel(), self.weight_map.ravel(), self.face_map,
            self.z_map.ravel(), lock,
        )

        # texture sampling
        background_colors = xp.array(self.background_color, 'float32')
        chainer.cuda.elementwise(
            'int32 pi, raw float32 textures, raw float32 face_map, int32 face_index, raw float32 weight_map, ' +
            'float32 z, raw float32 background_color',
            'raw float32 images, raw float32 sampling_weight_map, raw int32 sampling_index_map',
            string.Template('''
                int is = ${image_size};
                int nf = ${num_faces};
                int ts = ${texture_size};
                int bn = pi / (is * is);

                float* pixel = &images[pi * 3];
                if (0 <= face_index) {
                    float* face = &face_map[pi * 9];
                    float* weight = &weight_map[pi * 3];
                    float* texture = &textures[(bn * nf + face_index) * ts * ts * ts * 3];
                    float new_pixel[3] = {0, 0, 0};

                    /* get texture index (float) */
                    float texture_index_float[3];
                    for (int k = 0; k < 3; k++) {
                        texture_index_float[k] = weight[k] * (ts - 1 - ${eps}) * (z / (face[2 + 3 * k]));
                    }
                    float* sampling_weight_map_p = &sampling_weight_map[pi * 8];
                    int* sampling_index_map_p = &sampling_index_map[pi * 8];
                    for (int pn = 0; pn < 8; pn++) {
                        /* blend */
                        float w = 1;                        // weight
                        int texture_index_int[3];            // index in source (int)
                        for (int k = 0; k < 3; k++) {
                            if ((pn >> k) % 2 == 0) {
                                w *= 1 - (texture_index_float[k] - (int)texture_index_float[k]);
                                texture_index_int[k] = (int)texture_index_float[k];
                            } else {
                                w *= texture_index_float[k] - (int)texture_index_float[k];
                                texture_index_int[k] = (int)texture_index_float[k] + 1;
                            }
                        }

                        int isc = texture_index_int[0] * ts * ts + texture_index_int[1] * ts + texture_index_int[2];
                        for (int k = 0; k < 3; k++) new_pixel[k] += w * texture[isc * 3 + k];
                        *sampling_weight_map_p++ = w;
                        *sampling_index_map_p++ = isc;
                    }
                    memcpy(pixel, new_pixel, 3 * sizeof(float));
                } else {
                    for (int k = 0; k < 3; k++) pixel[k] = background_color[k];
                }
            ''').substitute(
                image_size=is_,
                num_faces=nf,
                texture_size=ts,
                eps=self.eps,
            ),
            'function',
        )(
            xp.arange(bs * is_ * is_).astype('int32'), textures, self.face_map, self.face_index_map.ravel(),
            self.weight_map, self.z_map.ravel(), background_colors,
            self.images, self.sampling_weight_map, self.sampling_index_map,
        )
        return self.images,

    def backward_gpu(self, inputs, grad_outputs):
        xp = chainer.cuda.get_array_module(inputs[0])
        faces = xp.ascontiguousarray(inputs[0])
        textures = xp.ascontiguousarray(inputs[1])
        grad_images = xp.ascontiguousarray(grad_outputs[0])
        grad_faces = xp.ascontiguousarray(xp.zeros_like(faces, dtype='float32'))
        grad_textures = xp.ascontiguousarray(xp.zeros_like(textures, dtype='float32'))
        bs, nf = faces.shape[:2]
        is_ = self.image_size
        ts = textures.shape[2]

        # backward texture sampling
        chainer.cuda.elementwise(
            'int32 pi, int32 face_index, raw T sampling_weight_map, raw int32 sampling_index_map, raw T grad_images, ' +
            'raw T grad_textures',
            '',
            string.Template('''
                int is = ${image_size};
                int nf = ${num_faces};
                int ts = ${texture_size};
                int bn = pi / (is * is);    // batch number [0 -> bs]

                if (0 <= face_index) {
                    float* grad_texture = &grad_textures[(bn * nf + face_index) * ts * ts * ts * 3];
                    float* sampling_weight_map_p = &sampling_weight_map[pi * 8];
                    int* sampling_index_map_p = &sampling_index_map[pi * 8];
                    for (int pn = 0; pn < 8; pn++) {
                        float w = *sampling_weight_map_p++;
                        int isc = *sampling_index_map_p++;
                        float* grad_texture_p = &grad_texture[isc * 3];
                        float* grad_images_p = &grad_images[pi * 3];
                        for (int k = 0; k < 3; k++) atomicAdd(grad_texture_p++, w * *grad_images_p++);
                    }
                }
            ''').substitute(
                image_size=is_,
                num_faces=nf,
                texture_size=ts,
            ),
            'function',
        )(
            xp.arange(bs * is_ * is_).astype('int32'), self.face_index_map.ravel(), self.sampling_weight_map,
            self.sampling_index_map, grad_images, grad_textures,
        )

        # pseudo gradient
        loop = xp.arange(bs * nf).astype('int32')
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
                            float* color_in;
                            float* color_out;
                            if (axis == 0) {
                                color_in = &images[3 * (bn * is * is + d1_in * is + d0)];
                                color_out = &images[3 * (bn * is * is + d1_out * is + d0)];
                            } else {
                                color_in = &images[3 * (bn * is * is + d0 * is + d1_in)];
                                color_out = &images[3 * (bn * is * is + d0 * is + d1_out)];
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
                                    images_p = &images[3 * (bn * is * is + d1_from * is + d0)] - 3 * is;
                                    grad_images_p = &grad_images[3 * (bn * is * is + d1_from * is + d0)] - 3 * is;
                                } else {
                                    images_p = &images[3 * (bn * is * is + d0 * is + d1_from)] - 3;
                                    grad_images_p = &grad_images[3 * (bn * is * is + d0 * is + d1_from)] - 3;
                                }
                                for (int d1 = d1_from; d1 <= d1_to; d1++) {
                                    float diff_grad = 0;
                                    if (axis == 0) {
                                        images_p += 3 * is;
                                        grad_images_p += 3 * is;
                                    } else {
                                        images_p += 3;
                                        grad_images_p += 3;
                                    }
                                    for (int k = 0; k < 3; k++) {
                                        diff_grad += (images_p[k] - color_in[k]) * grad_images_p[k];
                                    }
                                    if (diff_grad <= 0) continue;
                                    if (p1_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (p1_d0 - d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p0_i * 3 + (1 - axis)] -= diff_grad / dist;
                                    }
                                    if (p0_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (d0 - p0_d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p1_i * 3 + (1 - axis)] -= diff_grad / dist;
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
                                    images_p = &images[3 * (bn * is * is + d1_from * is + d0)] - 3 * is;
                                    grad_images_p = &grad_images[3 * (bn * is * is + d1_from * is + d0)] - 3 * is;
                                } else {
                                    face_index_map_p = &face_index_map[bn * is * is + d0 * is + d1_from] - 1;
                                    images_p = &images[3 * (bn * is * is + d0 * is + d1_from)] - 3;
                                    grad_images_p = &grad_images[3 * (bn * is * is + d0 * is + d1_from)] - 3;
                                }
                                for (int d1 = d1_from; d1 <= d1_to; d1++) {
                                    float diff_grad = 0;
                                    if (axis == 0) {
                                        face_index_map_p += is;
                                        images_p += 3 * is;
                                        grad_images_p += 3 * is;
                                        if (*face_index_map_p != fn) continue;
                                    } else {
                                        face_index_map_p += 1;
                                        images_p += 3;
                                        grad_images_p += 3;
                                        if (*face_index_map_p != fn) continue;
                                    }
                                    for (int k = 0; k < 3; k++) {
                                        diff_grad += (images_p[k] - color_out[k]) * grad_images_p[k];
                                    }
                                    if (diff_grad <= 0) continue;
                                    if (p1_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (p1_d0 - d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p0_i * 3 + (1 - axis)] -= diff_grad / dist;
                                    }
                                    if (p0_d0 != d0) {
                                        float dist = (p1_d0 - p0_d0) / (d0 - p0_d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[p1_i * 3 + (1 - axis)] -= diff_grad / dist;
                                    }
                                }
                            }
                        }
                    }
                }

                /* set to global gradient variable */
                for (int k = 0; k < 9; k++) grad_faces[i * 9 + k] = grad_face[k];
            ''').substitute(
                image_size=is_,
                num_faces=nf,
                eps=self.eps,
            ),
            'function',
        )(loop, faces, self.face_index_map, self.images, grad_images.ravel(), grad_faces)

        return grad_faces, grad_textures

    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError


def rasterize(faces, textures, image_size=256, anti_aliasing=True, near=0.1, far=100, eps=1e-3, background_color=(0, 0, 0)):
    if anti_aliasing:
        images = Rasterize(image_size * 2, near, far, eps, background_color)(faces, textures)
        images = images.transpose((0, 3, 1, 2))
        images = cf.average_pooling_2d(images, 2, 2)
    else:
        images = Rasterize(image_size, near, far, eps, background_color)(faces, textures)
        images = images.transpose((0, 3, 1, 2))
    images = images[:, :, ::-1, :]
    return images
