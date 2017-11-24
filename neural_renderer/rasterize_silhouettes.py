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
        inputs = [xp.ascontiguousarray(i) for i in inputs]
        faces, = inputs
        bs, nf = faces.shape[:2]
        is_ = self.image_size

        # initialize buffers
        self.face_index_map = xp.ascontiguousarray(-1 * xp.ones((bs, is_, is_), dtype='int32'))
        self.images = xp.ascontiguousarray(xp.zeros((bs, is_, is_), dtype='float32'))

        # vertices -> face_index_map, weight_map, z_map
        # face_index_map = -1 if background
        chainer.cuda.elementwise(
            'raw float32 faces, int32 num_faces, int32 image_size, float32 near, float32 far',
            'int32 face_index_map, float32 images',
            '''
                /* current position & index */
                const int nf = num_faces;                   // short name
                const int is = image_size;                  // short name
                const int is2 = is * is;                    // number of pixels
                const int pi = i;                           // pixel index on all batches
                const int bn = pi / (is2);                  // batch number
                const int yi = (pi % (is2)) / is;           // index of current y-position [0, is - 1]
                const int xi = (pi % (is2)) % is;           // index of current x-position [0, is - 1]
                const float yp = (1 - 1. / is) * ((2. / (is - 1)) * yi - 1);   // coordinate of y-position [-1, 1]
                const float xp = (1 - 1. / is) * ((2. / (is - 1)) * xi - 1);   // coordinate of x-position [-1, 1]

                /* for each face */
                float* face;            // current face
                float z_min = far;      // z of nearest face
                int z_min_fn = -1;      // face number of nearest face
                for (int fn = 0; fn < nf; fn++) {
                    /* go to next face */
                    if (fn == 0) {
                        face = &faces[(bn * nf) * 3 * 3];
                    } else {
                        face += 3 * 3;
                    }

                    /* get vertex of current face */
                    const float x[3] = {face[0], face[3], face[6]};
                    const float y[3] = {face[1], face[4], face[7]};
                    const float z[3] = {face[2], face[5], face[8]};

                    /* check too close & too far */
                    if (z[0] < 0 || z[1] < 0 || z[2] < 0) continue;
                    if (z_min < z[0] && z_min < z[1] && z_min < z[2]) continue;

                    /* check [yp, xp] is inside the face */
                    if (((yp - y[0]) * (x[1] - x[0]) < (xp - x[0]) * (y[1] - y[0])) ||
                        ((yp - y[1]) * (x[2] - x[1]) < (xp - x[1]) * (y[2] - y[1])) ||
                        ((yp - y[2]) * (x[0] - x[2]) < (xp - x[2]) * (y[0] - y[2]))) continue;

                    /* compute f_inv */
                    float f_inv[9] = {
                        y[1] - y[2], x[2] - x[1], x[1] * y[2] - x[2] * y[1],
                        y[2] - y[0], x[0] - x[2], x[2] * y[0] - x[0] * y[2],
                        y[0] - y[1], x[1] - x[0], x[0] * y[1] - x[1] * y[0]};
                    float f_inv_denominator = x[2] * (y[0] - y[1]) + x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]);
                    for (int k = 0; k < 9; k++) f_inv[k] /= f_inv_denominator;

                    /* compute w = f_inv * p */
                    float w[3];
                    for (int k = 0; k < 3; k++) w[k] = f_inv[3 * k + 0] * xp + f_inv[3 * k + 1] * yp + f_inv[3 * k + 2];

                    /* sum(w) -> 1, 0 < w < 1 */
                    float w_sum = 0;
                    for (int k = 0; k < 3; k++) {
                        if (w[k] < 0) w[k] = 0;
                        if (1 < w[k]) w[k] = 1;
                        w_sum += w[k];
                    }
                    if (1 < w_sum) for (int k = 0; k < 3; k++) w[k] /= w_sum;

                    /* compute 1 / zp = sum(w / z) & check z-buffer */
                    const float zp = 1. / (w[0] / z[0] + w[1] / z[1] + w[2] / z[2]);
                    if (zp <= near || far <= zp) continue;

                    /* check nearest */
                    if (zp < z_min) {
                        z_min = zp;
                        z_min_fn = fn;
                    }
                }
                /* set to buffer */
                if (0 <= z_min_fn) {
                    face_index_map = z_min_fn;
                    images = 1.;
                }
            ''',
            'function',
        )(faces, nf, is_, self.near, self.far, self.face_index_map.ravel(), self.images.ravel())

        return self.images,

    def backward_gpu(self, inputs, grad_outputs):
        raise NotImplementedError

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
