import math


def get_points_from_angles(distance, elevation, azimuth):
    if isinstance(distance, float) or isinstance(distance, int):
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        raise NotImplementedError
