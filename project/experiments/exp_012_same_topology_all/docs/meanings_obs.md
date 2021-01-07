# means of obs

Total 28 dimensions:

[more]
0: relative height of torso `z - self.initial_z`
1: yaw `sin(angle_to_target)`
2: yaw `cos(angle_to_target)`
3: velocity in x direction `0.3 * vx`
4: velocity in y direction `0.3 * vy`
5: velocity in z direction `0.3 * vz`
6: roll
7: pitch

[joints_position_value]
8: `front_left_hip` position `(2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit)`
9: `front_left_hip` changing velocity `0.1 * vel`
10: `front_left_ankle` position
11: `front_left_ankle` changing velocity
12: `front_right_hip` position
13: `front_right_hip` changing velocity
14: `front_right_ankle` position
15: `front_right_ankle` changing velocity
16: `back_left_hip` position
17: `back_left_hip` changing velocity
18: `back_left_ankle` position
19: `back_left_ankle` changing velocity
20: `back_right_hip` position
21: `back_right_hip` changing velocity
22: `back_right_ankle` position
23: `back_right_ankle` changing velocity

[feet_contact]
24: does `front_left_foot` contact with the floor 
25: `front_right_foot`
26: `left_back_foot`
27: `right_back_foot`