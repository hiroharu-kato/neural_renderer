import chainer
import chainer.functions as cf


def perspective(vertices, angle=None):
    assert (vertices.ndim == 3)
    if angle is None:
        xp = chainer.cuda.get_array_module(vertices)
        angle = xp.array([xp.radians(60)] * vertices.shape[0], 'float32')

    width = cf.tan(angle / 2)
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width[:, None]
    y = vertices[:, :, 1] / z / width[:, None]
    vertices = cf.concat((x[:, :, None], y[:, :, None], z[:, :, None]), axis=2)
    return vertices
