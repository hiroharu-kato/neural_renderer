import chainer
import chainer.functions as cf
import neural_renderer

def look_at(vertices, eye_from, at=None, up=None):
    """
    "Look at" transformation of vertices.
    Based on http://marina.sys.wakayama-u.ac.jp/~tokoi/?date=20090902.
    """
    assert (vertices.ndim == 4)
    bs, nf = vertices.shape[:2]

    xp = chainer.cuda.get_array_module(vertices)
    if at is None:
        at = xp.array([0, 0, 0], 'float32')
    if up is None:
        up = xp.zeros([0, 1, 0], 'float32')

    if eye_from.ndim == 1:
        eye_from = eye_from[None, :]
    if at.ndim == 1:
        at = at[None, :]
    if up.ndim == 1:
        up = up[None, :]

    # create new axes
    z_axis = cf.normalize(at - eye_from)
    x_axis = cf.normalize(neural_renderer.cross(z_axis, up))
    y_axis = cf.normalize(neural_renderer.cross(x_axis, z_axis))

    # create rotation matrix: [bs, 3, 3]
    r = xp.concatenate((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), axis=1)
    if r.shape[0] != vertices.shape[0]:
        r = cf.broadcast_to(r, vertices.shape)

    # apply
    # [bs, nf, 3, 3] -> [bs, nf, 3, 3] -> [bs, nf, 3, 3]
    vertices -= eye_from[:, None, None, :]
    vs = vertices.shape
    vertices = cf.sum(
        cf.broadcast_to(vertices[:, :, :, None, :], (vs[0], vs[1], vs[2], 3, vs[3])) *
        r[:, None, None, :, :], axis=-1)

    return vertices



