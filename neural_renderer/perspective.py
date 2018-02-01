import chainer
import chainer.functions as cf


def perspective(vertices, angle=30):
    assert (vertices.ndim == 3)
    xp = chainer.cuda.get_array_module(vertices)
    angle = xp.array([xp.radians(angle)] * vertices.shape[0], 'float32')

    width = cf.tan(angle)
    width = cf.broadcast_to(width[:, None], vertices.shape[:2])
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = cf.concat((x[:, :, None], y[:, :, None], z[:, :, None]), axis=2)
    return vertices
