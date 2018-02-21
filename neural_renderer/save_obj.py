import os


def save_obj(filename, vertices, faces):
    assert vertices.ndim == 2
    assert faces.ndim == 2

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')
        f.write('g mesh\n')
        f.write('\n')
        for vertex in vertices:
            f.write('v  %.4f %.4f %.4f\n' % (vertex[0], vertex[1], vertex[2]))
        f.write('\n')
        for face in faces:
            f.write('f  %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))
