"""
Example 4. Finding camera parameters.
"""
import argparse
import glob
import os
import subprocess

import chainer
import chainer.functions as cf
import numpy as np
import scipy.misc
import tqdm

import neural_renderer


class Model(chainer.Link):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        with self.init_scope():
            # load .obj
            vertices, faces = neural_renderer.load_obj(filename_obj)
            self.vertices = vertices[None, :, :]
            self.faces = faces[None, :, :]

            # create textures
            texture_size = 2
            textures = np.ones((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3), 'float32')
            self.textures = textures

            # load reference image
            self.image_ref = (scipy.misc.imread(filename_ref).max(-1) != 0).astype('float32')

            # camera parameters
            self.camera_position = chainer.Parameter(np.array([6, 10, -14], 'float32'))

            # setup renderer
            renderer = neural_renderer.Renderer()
            renderer.eye = self.camera_position
            self.renderer = renderer

    def to_gpu(self):
        super(Model, self).to_gpu()
        self.faces = chainer.cuda.to_gpu(self.faces)
        self.vertices = chainer.cuda.to_gpu(self.vertices)
        self.textures = chainer.cuda.to_gpu(self.textures)
        self.image_ref = chainer.cuda.to_gpu(self.image_ref)

    def __call__(self):
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = cf.sum(cf.square(image - self.image_ref[None, :, :]))
        return loss


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-ir', '--filename_ref', type=str, default='./examples/data/example4_ref.png')
    parser.add_argument('-or', '--filename_output', type=str, default='./examples/data/example4_result.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    model = Model(args.filename_obj, args.filename_ref)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer.setup(model)
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        optimizer.target.cleargrads()
        loss = model()
        loss.backward()
        optimizer.update()
        images = model.renderer.render(model.vertices, model.faces, cf.tanh(model.textures))
        image = images.data.get()[0]
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, i))
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.data < 70:
            break
    make_gif(working_directory, args.filename_output)


if __name__ == '__main__':
    run()
