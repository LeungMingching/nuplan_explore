import math
import numpy as np

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    # in radians, [-pi, pi]
    return roll_x, pitch_y, yaw_z

def find_nearest_idx(array, value):
    distance = np.absolute(array - value*np.ones_like(array))
    return np.argmin(distance)

if __name__ == '__main__':
    array = np.array([0,1,3,4,5,6,67])
    value = 66

    a = find_nearest_idx(array, value)
    print(a)