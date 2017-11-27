import chainer
import chainer.functions as cf
import chainer.gradient_check
import chainer.testing
import cupy as cp
import numpy as np
import scipy.misc
import unittest

import neural_renderer


class TestRasterizeSilhouettes(unittest.TestCase):
    def test_case1(self):
        # load teapot
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices = vertices[None, :, :]
        faces = faces[None, :, :]
        vertices = chainer.cuda.to_gpu(vertices)
        faces = chainer.cuda.to_gpu(faces)

        # create renderer
        renderer = neural_renderer.Renderer()
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer.render_silhouettes(vertices, faces)
        images = images.data.get()
        image = images[0]

        # load reference image by blender
        ref = scipy.misc.imread('./tests/data/teapot_blender.png')
        ref = ref.astype('float32')
        ref = (ref.min(-1) != 255).astype('float32')

        chainer.testing.assert_allclose(ref, image)

    def test_backward_case1(self):
        vertices = [
            [0.8, 0.8, 1.],
            [0.0, -0.5, 1.],
            [0.2, -0.4, 1.]]
        faces = [[0, 1, 2]]
        pxi = 35
        pyi = 25

        renderer = neural_renderer.Renderer()
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False

        vertices = chainer.Variable(cp.array(vertices, 'float32'))
        faces = cp.array(faces, 'int32')
        images = renderer.render_silhouettes(vertices[None, :, :], faces[None, :, :])
        loss = cf.sum(cf.absolute(images[:, pyi, pxi] - 1))
        loss.backward()

        for i in range(3):
            for j in range(2):
                axis = 'x' if j == 0 else 'y'
                vertices2 = cp.copy(vertices.data)
                vertices2[i, j] -= 1. / vertices.grad[i, j]
                images = renderer.render_silhouettes(vertices2[None, :, :], faces[None, :, :])
                image = np.tile(images[0].data.get()[:, :, None], (1, 1, 3))
                image[pyi, pxi] = [1, 0, 0]
                ref = scipy.misc.imread('./tests/data/rasterize_silhouettes_case1_v%d_%s.png' % (i, axis))
                ref = ref.astype('float32') / 255
                chainer.testing.assert_allclose(ref, image)

    def test_backward_case2(self):
        vertices = [
            [0.8, 0.8, 1.],
            [-0.5, -0.8, 1.],
            [0.8, -0.8, 1.]]
        faces = [[0, 1, 2]]
        pyi = 40
        pxi = 50

        renderer = neural_renderer.Renderer()
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False

        vertices = chainer.Variable(cp.array(vertices, 'float32'))
        faces = cp.array(faces, 'int32')
        images = renderer.render_silhouettes(vertices[None, :, :], faces[None, :, :])
        loss = cf.sum(cf.absolute(images[:, pyi, pxi]))
        loss.backward()

        for i in range(3):
            for j in range(2):
                axis = 'x' if j == 0 else 'y'
                vertices2 = cp.copy(vertices.data)
                vertices2[i, j] -= 1. / vertices.grad[i, j]
                images = renderer.render_silhouettes(vertices2[None, :, :], faces[None, :, :])
                image = np.tile(images[0].data.get()[:, :, None], (1, 1, 3))
                image[pyi, pxi] = [1, 0, 0]
                ref = scipy.misc.imread('./tests/data/rasterize_silhouettes_case2_v%d_%s.png' % (i, axis))
                ref = ref.astype('float32') / 255
                chainer.testing.assert_allclose(ref, image)


if __name__ == '__main__':
    unittest.main()
