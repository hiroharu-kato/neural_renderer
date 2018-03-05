import string

import chainer
import numpy as np
import skimage.io


def load_textures(filename_obj, filename_texture, texture_size):
    """
    WARNING: this function is not well tested.
    """

    # load vertices
    vertices = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype('float32')

    # load faces for textures
    faces = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[1])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[1])
                v2 = int(vs[i + 2].split('/')[1])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype('int32') - 1
    faces = vertices[faces]
    faces = chainer.cuda.to_gpu(faces)
    faces = faces % 1

    textures = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
    textures = chainer.cuda.to_gpu(textures)
    image = skimage.io.imread(filename_texture).astype('float32') / 255.
    image = chainer.cuda.to_gpu(image)
    image = image[::-1, ::]

    loop = np.arange(textures.size / 3).astype('int32')
    loop = chainer.cuda.to_gpu(loop)
    chainer.cuda.elementwise(
        'int32 j, raw float32 image, raw float32 faces, raw float32 textures',
        '',
        string.Template('''
            const int ts = ${texture_size};
            const int fn = i / (ts * ts * ts);
            float dim0 = ((i / (ts * ts)) % ts) / (ts - 1.) ;
            float dim1 = ((i / ts) % ts) / (ts - 1.);
            float dim2 = (i % ts) / (ts - 1.);
            if (1 < dim0 + dim1 + dim2) {
                float sum = dim0 + dim1 + dim2;
                dim0 /= sum;
                dim1 /= sum;
                dim2 /= sum;
            }
            const float* face = &faces[fn * 3 * 2];
            float* texture = &textures[i * 3];

            const float pos_x = (
                (face[2 * 0 + 0] * dim0 + face[2 * 1 + 0] * dim1 + face[2 * 2 + 0] * dim2) * (${image_width} - 1));
            const float pos_y = (
                (face[2 * 0 + 1] * dim0 + face[2 * 1 + 1] * dim1 + face[2 * 2 + 1] * dim2) * (${image_height} - 1));
            const float weight_x1 = pos_x - (int)pos_x;
            const float weight_x0 = 1 - weight_x1;
            const float weight_y1 = pos_y - (int)pos_y;
            const float weight_y0 = 1 - weight_y1;
            for (int k = 0; k < 3; k++) {
                float c = 0;
                c += image[((int)pos_y * ${image_width} + (int)pos_x) * 3 + k] * (weight_x0 * weight_y0);
                c += image[((int)(pos_y + 1) * ${image_width} + (int)pos_x) * 3 + k] * (weight_x0 * weight_y1);
                c += image[((int)pos_y * ${image_width} + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y0);
                c += image[((int)(pos_y + 1)* ${image_width} + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y1);
                texture[k] = c;
            }
        ''').substitute(
            texture_size=texture_size,
            image_height=image.shape[0],
            image_width=image.shape[1],
        ),
        'function',
    )(loop, image, faces, textures)
    return textures.get()


def load_obj(filename_obj, normalization=True, filename_texture=None, texture_size=4):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype('float32')

    # load faces
    faces = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype('int32') - 1

    # load textures
    if filename_texture is not None:
        textures = load_textures(filename_obj, filename_texture, texture_size)
    else:
        textures = None

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[None, :] / 2

    if textures is None:
        return vertices, faces
    else:
        return vertices, faces, textures
