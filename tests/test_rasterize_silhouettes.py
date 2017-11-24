import unittest

import chainer
import chainer.gradient_check
import chainer.testing
import numpy as np

import neural_renderer


class TestRasterizeSilhouette(unittest.TestCase):
    def test_case1(self):

        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices = vertices[None, :, :]
        faces = faces[None, :, :]

        vertices = neural_renderer.look_at(vertices, np.array([0, 0, -2.732], 'float32'))
        vertices = neural_renderer.perspective(vertices)

        vertices = neural_renderer.vertices_to_faces(vertices, faces)
        print vertices.shape


if __name__ == '__main__':
    unittest.main()
