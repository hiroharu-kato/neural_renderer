import argparse
import os
import chainer
import numpy as np
import scipy.misc

import neural_renderer


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fi', '--filename_input', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-fo', '--filename_output', type=str, default='./examples/data/example1.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2

    # load .obj
    vertices, faces = neural_renderer.load_obj(args.filename_input)
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]

    # create texture
    textures = np.ones((1, faces.shape[1], texture_size, texture_size, texture_size, 3), 'float32')

    # to gpu
    chainer.cuda.get_device_from_id(args.gpu).use()
    vertices = chainer.cuda.to_gpu(vertices)
    faces = chainer.cuda.to_gpu(faces)
    textures = chainer.cuda.to_gpu(textures)

    # create renderer
    renderer = neural_renderer.Renderer()

    for azimuth in range(0, 360, 4):
        renderer.eye = neural_renderer.get_points_from_angles(camera_distance, elevation, azimuth)
        images = renderer.render(vertices, faces, textures)
        image = images.data.get()[0].transpose((1, 2, 0))
        scipy.misc.imsave('%s/_tmp_%04d.png' % os.path.dirname(args.filename_output), image)


if __name__ == '__main__':
    run()
