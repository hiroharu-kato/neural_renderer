import math

import chainer


def get_points_from_angles(distance, elevation, azimuth, digrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if digrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        xp = chainer.cuda.get_array_module(distance)
        if digrees:
            elevation = xp.radians(elevation)
            azimuth = xp.radians(azimuth)
        return xp.stack([
            distance * xp.cos(elevation) * xp.sin(azimuth),
            distance * xp.sin(elevation),
            -distance * xp.cos(elevation) * xp.cos(azimuth),
        ]).transpose()
