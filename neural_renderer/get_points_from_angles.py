import math


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
        raise NotImplementedError
