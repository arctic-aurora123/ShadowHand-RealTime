import os
import numpy as np
import os.path as osp
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import pyrfuniverse.attributes as attr
import update
import math

# Initialize the environment
env = RFUniverseBaseEnv()

## the second element of the tuple is 0->x, 1->y, 2->z
read_list = [(0,0),(0,1),     (13,0),(13,1),(14,1),(14,2),(15,1),
        (0,2),    (7,0),(7,1),(8,1),(9,1),   (10,0),(10,1),(11,1),(12,1),   (4,0),(4,1),(5,1),(6,1),   (1,0),(1,1),(2,1),(3,1)]
## the seq of fingers: wrist,  thumb,  littlefinger, ringfinger, middlefinger, forefinger 
## each finger joint is from bottom to top

shadow = env.LoadURDF(path=os.path.abspath(osp.join('gripper', 'shadow.urdf')), native_ik=False)

shadow.SetTransform(position=[0, 1, 0], rotation=[0, 90, 0])
env.SetViewTransform(position=[0, 1.3, -0.4])
env.step(1)
env.ViewLookAt((np.asarray(shadow.data['position']) + np.asarray([0, 0.3, 0])).tolist())
env.step(10)
moveable_joint_count = shadow.data["number_of_moveable_joints"]
upper_limit = np.asarray(shadow.data['joint_upper_limit'])

idx = np.asarray([0,1,2,3,4,5,6,7,8,12,16])
angle, set_angle = np.zeros(moveable_joint_count), np.zeros(moveable_joint_count)

while True:
    rot = update.update_data()

    for m in range(24):
        if m in idx:
            set_angle[m] = 0
        else:
#            if abs(rot[read_list[m]]) <= upper_limit[m]: ## limits need to add
                set_angle[m] = rot[read_list[m]]
        if m == 8:
            set_angle[m] -= 20
        if m == 12:
            set_angle[m] -= 5
        if m == 16:
            set_angle[m] -= 5
        if m == 20:
            set_angle[m] -= 15
#            else:
#                set_angle[m] = upper_limit[m]
     # use this function to set joint parameter of shadow hand, the joint angle is described in degree measure
        shadow.SetJointPosition(set_angle.tolist())
        print(set_angle)
        # use env.step() to update the hand rendered in the simulation
        env.step(1)


