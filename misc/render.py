import math
import os
import random

import bpy
import scipy.misc


IMAGE_SIZE = 64
DISTANCE = 2.732


def set_camera_location(elevation, azimuth, distance):
    # set location
    x = 1 * math.cos(math.radians(-azimuth)) * math.cos(math.radians(elevation)) * distance
    y = 1 * math.sin(math.radians(-azimuth)) * math.cos(math.radians(elevation)) * distance
    z = 1 * math.sin(math.radians(elevation)) * distance
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    # look at center
    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def render(directory, elevation=30, distance=DISTANCE):
    for azimuth in range(0, 360, 15):
        filename = os.path.join(directory, 'e%03d_a%03d.png' % (elevation, azimuth))
        set_camera_location(elevation, azimuth, distance)
        bpy.context.scene.render.filepath = filename
        bpy.ops.render.render(write_still=True)

        if False:
            img = scipy.misc.imread(filename)[:, :, :].astype('float32') / 255.
            if False:
                img = (img[::2, ::2] + img[1::2, ::2] + img[::2, 1::2] + img[1::2, 1::2]) / 4.
            else:
                import chainer.functions as cf
                img = img.transpose((2, 0, 1))[None, :, :, :]
                img = cf.resize_images(img, (64, 64))
                img = img[0].data.transpose((1, 2, 0))

            img = (img * 255).clip(0., 255.).astype('uint8')
            scipy.misc.imsave(filename, img)


def setup():
    context = bpy.context
    if False:
        context.scene.render.resolution_x = IMAGE_SIZE * 2
        context.scene.render.resolution_y = IMAGE_SIZE * 2
        context.scene.render.resolution_percentage = 100
        context.scene.render.use_antialiasing = False
    else:
        context.scene.render.resolution_x = IMAGE_SIZE
        context.scene.render.resolution_y = IMAGE_SIZE
        context.scene.render.resolution_percentage = 100
        context.scene.render.use_antialiasing = True
    context.scene.render.use_free_unused_nodes = True
    context.scene.render.use_free_image_textures = True
    context.scene.render.alpha_mode = 'TRANSPARENT'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # camera
    camera = bpy.data.cameras.values()[0]
    camera.sensor_width = 1
    camera.sensor_height = 1
    camera.lens = 1.8660254037844388

    # lighting
    light = bpy.data.objects['Lamp']
    light.data.energy = 1
    context.scene.world.light_settings.use_environment_light = True
    context.scene.world.light_settings.environment_energy = 0.5
    context.scene.world.light_settings.environment_color = 'PLAIN'


def load_obj(filename):
    # filename = '/home/hkato/temp/obj/model.obj'
    # filename = '/media/disk2/lab/large_data/ShapeNetCore.v1/03001627/1bcec47c5dc259ea95ca4adb70946a21/model.obj'
    bpy.ops.import_scene.obj(filepath=filename, use_smooth_groups=False, use_split_objects=False,
                             use_split_groups=False)
    object_id = len(bpy.data.objects) - 1
    obj = bpy.data.objects[object_id]
    bpy.context.scene.objects.active = obj

    # get max & min of vertices
    inf = 10000
    vertex_max = [-inf, -inf, -inf]
    vertex_min = [inf, inf, inf]
    for j in range(8):
        for i in range(3):
            vertex_max[i] = max(vertex_max[i], obj.bound_box[j][i])
            vertex_min[i] = min(vertex_min[i], obj.bound_box[j][i])
    dimensions = obj.dimensions  # = max - min

    # centering
    for i in range(3):
        obj.location[i] += (vertex_max[i] + vertex_min[i]) / 2

    # scaling
    scale = max(dimensions)
    for i in range(3):
        obj.scale[i] = obj.scale[i] / scale

    # materials
    for m in bpy.data.materials:
        m.ambient = 0.5
        m.use_shadeless = False
        m.use_transparency = False
        m.use_raytrace = False


def clear():
    bpy.ops.wm.open_mainfile(filepath='/home/hkato/temp/untitled.blend')


def run():
    class_ids = [
        '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263',
        '04256520', '04379243', '04401088', '04530566']
    # class_ids = ['03001627']
    directory_shapenet_id = '../../resource/shapenetcore_ids'
    directory_rendering = '/media/disk2/lab/projection/reconstruction/shapenet_images_%d_%.1f/%s/%s'
    filename_shapenet_obj = '/media/disk2/lab/large_data/ShapeNetCore.v1/%s/%s/model.obj'

    # ce33bf3ec6438e5bef662d1962a11f02
    for class_id in class_ids:
        ids = open(os.path.join(directory_shapenet_id, '%s_trainids.txt' % class_id)).readlines()
        ids += open(os.path.join(directory_shapenet_id, '%s_valids.txt' % class_id)).readlines()
        ids += open(os.path.join(directory_shapenet_id, '%s_testids.txt' % class_id)).readlines()
        obj_ids = [i.strip().split('/')[-1] for i in ids if len(i.strip()) != 0]

        for i, obj_id in enumerate(obj_ids):
            print('rendering: %s %d / %d' % (class_id, i, len(obj_ids)))

            directory = directory_rendering % (IMAGE_SIZE, DISTANCE, class_id, obj_id)
            directory_tmp = directory + '_'
            if os.path.exists(directory):
                continue
            if os.path.exists(directory_tmp):
                continue
            try:
                os.makedirs(directory_tmp)
            except:
                continue

            clear()
            setup()
            load_obj(filename_shapenet_obj % (class_id, obj_id))
            render(directory_tmp)
            try:
                os.rename(directory_tmp, directory)
            except:
                continue


run()
