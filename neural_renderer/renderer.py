import neural_renderer
import chainer.functions as cf


class Renderer(object):
    def __init__(self):
        self.image_size = 256
        self.anti_aliasing = True
        self.perspective = True
        self.focal_length = 1.732
        self.eye = [0, 0, -self.focal_length - 1]
        self.background_color = [0, 0, 0]
        self.near = 0.1
        self.far = 100
        self.rasterizer_eps = 1e-3
        self.fill_back = True

        # light
        self.light_intensity_ambient = 0.5
        self.light_intensity_directional = 0.5

    def render_silhouettes(self, vertices, faces):
        vertices = neural_renderer.look_at(vertices, self.eye)
        if self.perspective:
            vertices = neural_renderer.perspective(vertices)
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1)
        images = neural_renderer.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render(self, vertices, faces, textures):
        vertices = neural_renderer.look_at(vertices, self.eye)
        if self.perspective:
            vertices = neural_renderer.perspective(vertices)
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1)
            textures = cf.concat((textures, textures.transpose((0, 1, 4, 3, 2, 5))), axis=1)
        textures = neural_renderer.lighting(
            faces, textures, self.light_intensity_ambient, self.light_intensity_directional)
        images = neural_renderer.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images