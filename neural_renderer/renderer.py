import neural_renderer


class Renderer(object):
    def __init__(self):
        self.image_size = 256
        self.anti_aliasing = True
        self.perspective = True
        self.focal_length = 1.732
        self.eye = [0, 0, -self.focal_length - 1]

    def render_silhouettes(self, vertices, faces):
        vertices = neural_renderer.look_at(vertices, self.eye)
        if self.perspective:
            vertices = neural_renderer.perspective(vertices)
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        images = neural_renderer.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_silhouettes(self, vertices, faces, textures):
        vertices = neural_renderer.look_at(vertices, self.eye)
        if self.perspective:
            vertices = neural_renderer.perspective(vertices)
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        faces = neural_renderer.lighting(faces, textures)
        images = neural_renderer.rasterize(faces, textures, self.image_size, self.anti_aliasing)
        return images
