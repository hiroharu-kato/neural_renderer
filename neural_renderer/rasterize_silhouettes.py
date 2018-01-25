import chainer
import chainer.functions as cf
import string

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
        self.face_index_map = xp.ascontiguousarray(-1 * xp.ones((batch_size, image_size, image_size), dtype='int32'))
        self.images = xp.ascontiguousarray(xp.zeros((batch_size, image_size, image_size), dtype='float32'))

        # vertices -> face_index_map, z_map
        # face_index_map = -1 if background
        # fast implementation using unsafe pseudo mutex
        loop = xp.arange(batch_size * num_faces).astype('int32')
        lock = xp.zeros(self.images.shape, 'int32')
        z_buffer = xp.zeros(self.images.shape, 'float32') + self.far
        chainer.cuda.elementwise(
            'int32 j, raw float32 faces, raw int32 face_index_map, raw float32 images, raw float32 z_buffer, ' +
            'raw int32 lock',
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
                float p0_xi, p0_yi, p0_zp, p1_xi, p1_yi, p1_zp, p2_xi, p2_yi, p2_zp;
                if (face[0] < face[3]) {
                    if (face[6] < face[0]) p0_i = 2; else p0_i = 0;
                    if (face[3] < face[6]) p2_i = 2; else p2_i = 1;
                } else {
                    if (face[6] < face[3]) p0_i = 2; else p0_i = 1;
                    if (face[0] < face[6]) p2_i = 2; else p2_i = 0;
                }
                for (int k = 0; k < 3; k++) if (p0_i != k && p2_i != k) p1_i = k;
                p0_xi = (face[3 * p0_i + 0] * (1. * is) / (is - 1.) + 1) * (is - 1.) / 2.;
                p0_yi = (face[3 * p0_i + 1] * (1. * is) / (is - 1.) + 1) * (is - 1.) / 2.;
                p1_xi = (face[3 * p1_i + 0] * (1. * is) / (is - 1.) + 1) * (is - 1.) / 2.;
                p1_yi = (face[3 * p1_i + 1] * (1. * is) / (is - 1.) + 1) * (is - 1.) / 2.;
                p2_xi = (face[3 * p2_i + 0] * (1. * is) / (is - 1.) + 1) * (is - 1.) / 2.;
                p2_yi = (face[3 * p2_i + 1] * (1. * is) / (is - 1.) + 1) * (is - 1.) / 2.;
                p0_zp = face[3 * p0_i + 2];
                p1_zp = face[3 * p1_i + 2];
                p2_zp = face[3 * p2_i + 2];
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
                int xi_min = max(ceil(p0_xi), 0.);
                int xi_max = min(p2_xi, is - 1.);
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
                        while (atomicAdd(&lock[index], 0) != 0) {}
                        if (zp < z_buffer[index]) {
                            atomicExch(&images[index], 1);
                            atomicExch(&face_index_map[index], fn);
                            atomicExch(&z_buffer[index], zp);
                        }
                        atomicExch(&lock[index], 0);
                    }
                }
            ''').substitute(
                num_faces=num_faces,
                image_size=image_size,
                near=self.near,
                far=self.far,
            ),
            'function',
        )(loop, faces.ravel(), self.face_index_map.ravel(), self.images.ravel(), z_buffer, lock)

        return self.images,

    def backward_gpu(self, inputs, grad_outputs):
        xp = chainer.cuda.get_array_module(inputs[0])
        faces = xp.ascontiguousarray(inputs[0])
        grad_images = xp.ascontiguousarray(grad_outputs[0])
        grad_faces = xp.ascontiguousarray(xp.zeros_like(faces, dtype='float32'))
        num_faces = faces.shape[1]
        image_size = self.image_size

        # pseudo gradient
        chainer.cuda.elementwise(
            'raw float32 faces, raw int32 face_index_map, raw float32 images, float32 grad_images, ' +
            'raw float32 grad_faces',
            '',
            string.Template('''
                /* exit if no gradient from upper layers */
                if (grad_images == 0) return;

                /* compute current position & index */
                const int nf = ${num_faces};
                const int is = ${image_size};
                const int is2 = is * is;                    // number of pixels
                const int pi = i;                           // pixel index on all batches
                const int bn = pi / (is2);                  // batch number
                const int pyi = (pi % (is2)) / is;          // index of current y-position [0, is]
                const int pxi = (pi % (is2)) % is;          // index of current x-position [0, is]
                const float py = (1 - 1. / is) * ((2. / (is - 1)) * pyi - 1);   // coordinate of y-position [-1, 1]
                const float px = (1 - 1. / is) * ((2. / (is - 1)) * pxi - 1);   // coordinate of x-position [-1, 1]

                const int pfn = face_index_map[pi];        // face number of current position
                const float pp = images[pi];                // pixel intensity of current position

                for (int axis = 0; axis < 2; axis++) {
                    for (int direction = -1; direction <= 1; direction += 2) {
                        int qfn_last = pfn;
                        for (int d_pq = 1; d_pq < is; d_pq++) {
                            int qxi, qyi;
                            float qx, qy;
                            if (axis == 0) {
                                qxi = pxi + direction * d_pq;
                                qyi = pyi;
                                qx = (1 - 1. / is) * ((2. / (is - 1)) * qxi - 1);
                                qy = py;
                                if (qxi < 0 || is <= qxi) break;
                            } else {
                                qxi = pxi;
                                qyi = pyi + direction * d_pq;
                                qx = px;
                                qy = (1 - 1. / is) * ((2. / (is - 1)) * qyi - 1);
                                if (qyi < 0 || is <= qyi) break;
                            }

                            const int qi = bn * is2 + qyi * is + qxi;
                            const float qp = images[qi];
                            const float diff = qp - pp;
                            const int qfn = face_index_map[qi];

                            if (diff == 0) continue;                    // continue if same pixel value
                            if (0 <= diff * grad_images) continue;      // continue if wrong gradient
                            if (qfn == qfn_last) continue;              // continue if p & q are on same face

                            /* adjacent point to check edge */
                            int rxi, ryi;
                            float rx, ry;
                            if (axis == 0) {
                                rxi = qxi - direction;
                                ryi = pyi;
                                rx = (1 - 1. / is) * ((2. / (is - 1)) * rxi - 1);
                                ry = py;
                            } else {
                                rxi = pxi;
                                ryi = qyi - direction;
                                rx = px;
                                ry = (1 - 1. / is) * ((2. / (is - 1)) * ryi - 1);
                            }

                            for (int mode = 0; mode < 2; mode++) {
                                float* face;
                                float* grad_face;
                                if (mode == 0) {
                                    if (qfn < 0) continue;
                                    face = &faces[(bn * nf + qfn) * 3 * 3];
                                    grad_face = &grad_faces[(bn * nf + qfn) * 3 * 3];
                                } else if (mode == 1) {
                                    if (qfn_last != pfn) continue;
                                    if (pfn < 0) continue;
                                    face = &faces[(bn * nf + pfn) * 3 * 3];
                                    grad_face = &grad_faces[(bn * nf + pfn) * 3 * 3];
                                }

                                /* for each edge */
                                for (int vi0 = 0; vi0 < 3; vi0++) {
                                    /* get vertices */
                                    int vi1 = (vi0 + 1) % 3;
                                    float* v0 = &face[vi0 * 3];
                                    float* v1 = &face[vi1 * 3];

                                    /* get cross point */
                                    float sx, sy;
                                    if (axis == 0) {
                                        sx = (py - v0[1]) * (v1[0] - v0[0]) / (v1[1] - v0[1]) + v0[0];
                                        sy = py;
                                    } else {
                                        sx = px;
                                        sy = (px - v0[0]) * (v1[1] - v0[1]) / (v1[0] - v0[0]) + v0[1];
                                    }

                                    /* continue if not cross edge */
                                    if ((rx < sx) != (sx < qx)) continue;
                                    if ((ry < sy) != (sy < qy)) continue;
                                    if ((v0[1] < sy) != (sy < v1[1])) continue;
                                    if ((v0[0] < sx) != (sx < v1[0])) continue;

                                    /* signed distance (positive if pi < qi) */
                                    float dist_v0, dist_v1;
                                    if (axis == 0) {
                                        dist_v0 = (px - sx) * (v1[1] - v0[1]) / (v1[1] - py);
                                        dist_v1 = (px - sx) * (v0[1] - v1[1]) / (v0[1] - py);
                                    } else {
                                        dist_v0 = (py - sy) * (v1[0] - v0[0]) / (v1[0] - px);
                                        dist_v1 = (py - sy) * (v0[0] - v1[0]) / (v0[0] - px);
                                    }

                                    /* add small epsilon */
                                    dist_v0 = (0 < dist_v0) ? dist_v0 + ${eps} : dist_v0 - ${eps};
                                    dist_v1 = (0 < dist_v1) ? dist_v1 + ${eps} : dist_v1 - ${eps};

                                    /* gradient */
                                    const float g_v0 = grad_images * diff / dist_v0;
                                    const float g_v1 = grad_images * diff / dist_v1;

                                    atomicAdd(&grad_face[vi0 * 3 + axis], g_v0);
                                    atomicAdd(&grad_face[vi1 * 3 + axis], g_v1);
                                }
                            }
                            qfn_last = qfn;
                        }
                    }
                }
            ''').substitute(
                image_size=image_size,
                num_faces=num_faces,
                eps=self.eps,
            ),
            'function',
        )(faces, self.face_index_map, self.images, grad_images.ravel(), grad_faces)

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
