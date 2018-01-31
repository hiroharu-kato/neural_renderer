import unittest

import numpy as np

import neural_renderer


class TestCore(unittest.TestCase):
    def test_tetrahedron(self):
        vertices_ref = np.array(
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 0.]],
            'float32')
        faces_ref = np.array(
            [
                [1, 3, 2],
                [3, 1, 0],
                [2, 0, 1],
                [0, 2, 3]],
            'int32')

        vertices, faces = neural_renderer.load_obj('./tests/data/tetrahedron.obj', False)
        assert (np.allclose(vertices_ref, vertices))
        assert (np.allclose(faces_ref, faces))
        vertices, faces = neural_renderer.load_obj('./tests/data/tetrahedron.obj', True)
        assert (np.allclose(vertices_ref * 2 - 1.0, vertices))
        assert (np.allclose(faces_ref, faces))

    def test_teapot(self):
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        assert (faces.shape[0] == 2464)
        assert (vertices.shape[0] == 1292)


if __name__ == '__main__':
    unittest.main()
