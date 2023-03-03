import numpy as np
from scipy.spatial.transform import Rotation as R

class OrbitCamera:
    '''
        OrbitCamera class from ngp_pl: https://github.com/kwea123/ngp_pl/blob/master/show_gui.py
    '''
    def __init__(self, poses, r):
        self.radius = r
        self.center = np.zeros(3)
        # choose a pose as the initial rotation
        self.rot = poses[:3, :3]

        self.rotate_speed = 0.8
        self.res_defalut = poses
        self.params_changed = True

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def reset(self, pose=None):
        self.params_changed = True
        self.rot = np.eye(3)
        self.center = np.zeros(3)
        self.radius = 2.0
        if pose is not None:
            self.rot = pose.cpu().numpy()[:3, :3]

    def orbit(self, dx, dy):
        self.params_changed = True
        rotvec_x = self.rot[:, 1] * np.radians(100*self.rotate_speed  * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-100*self.rotate_speed  * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.params_changed = True
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.params_changed = True
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])