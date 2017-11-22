import chainer
import chainer.functions as cf
import numpy as np


def perspective(vertices, angle=np.pi / 3.):
    assert (vertices.ndim == 3)
    xp = chainer.cuda.get_array_module(vertices)

    width = xp.tan(angle / 2)
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = cf.concat((x[:, :, None], y[:, :, None], z[:, :, None]), axis=2)
    return vertices
