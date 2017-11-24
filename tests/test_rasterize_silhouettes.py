import numpy as np
import scipy.misc
import unittest

import chainer
import chainer.gradient_check
import chainer.testing

import neural_renderer


class TestRasterizeSilhouette(unittest.TestCase):
    def test_case1(self):
        """Comparison with blender."""
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices = vertices[None, :, :]
        faces = faces[None, :, :]
        vertices = chainer.cuda.to_gpu(vertices)
        faces = chainer.cuda.to_gpu(faces)

        eye = chainer.cuda.to_gpu(np.array([0, 0, -2.732], 'float32'))

        vertices = neural_renderer.look_at(vertices, eye)
        vertices = neural_renderer.perspective(vertices)
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        images = neural_renderer.rasterize_silhouettes(faces, anti_aliasing=False)
        images = images.data.get()
        image = images[0]

        ref = scipy.misc.imread('./tests/data/teapot_blender.png')
        ref = ref.astype('float32')
        ref = (ref.min(-1) != 255).astype('float32')

        chainer.testing.assert_allclose(ref, image)


if __name__ == '__main__':
    unittest.main()
