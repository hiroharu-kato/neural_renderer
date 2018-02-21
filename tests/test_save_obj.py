import os
import unittest
import numpy as np
import neural_renderer


class TestCore(unittest.TestCase):
    def test_save_obj(self):
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        neural_renderer.save_obj('./tests/data/teapot2.obj', vertices, faces)
        vertices2, faces2 = neural_renderer.load_obj('./tests/data/teapot.obj')
        os.remove('./tests/data/teapot2.obj')
        assert np.allclose(vertices, vertices2)
        assert np.allclose(faces, faces2)

if __name__ == '__main__':
    unittest.main()
