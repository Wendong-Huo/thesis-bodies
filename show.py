import time
from PyBulletWrapper.pybullet_wrapper.handy import HandyPyBullet
import pybullet as p
import pybullet_data
# import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=0)
args = parser.parse_args()
p = HandyPyBullet(p)
# p.connectPy()
p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
p.configureDebugVisualizer(flag=p.COV_ENABLE_GUI, enable=0, lightPosition=[1,-2,1])
p.setGravity(0,0,0)
p.resetDebugVisualizerCamera(1.5, 0, -20, [0,0,1])
objUid = p.loadMJCF(f"./dataset/walker2d_v6/bodies/{args.n}.xml")
objUid = objUid[0]
n = p.getNumJointsPy(bodyUniqueId=objUid)
for i in range(n):
    ret = p.getJointInfoPy(bodyUniqueId=objUid, jointIndex=i)
    print(i, ret["jointName"])
p.resetBasePositionAndOrientation(objUid, [0,0,0], [0,-0.12,0,1.])
jointIndices =      [6,     4,      12, ]
targetPositions =   [-0.3,  -0.3,   -0.3, ]
forces = [100. for j in jointIndices]
p.setJointMotorControlArrayPy(objUid, jointIndices=jointIndices, controlMode=p.POSITION_CONTROL, targetPositions=targetPositions, forces=forces)
for i in range(100):
    p.stepSimulation()

# viewMatrix = p.computeViewMatrix(
#     cameraEyePosition=[0, 0, 3],
#     cameraTargetPosition=[0, 0, 0],
#     cameraUpVector=[0, 1, 0])
# img = p.getCameraImagePy(width=2000, height=3000, viewMatrix=viewMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
# cv2.imwrite(f"{args.n}.png", img["rgbPixels"])

time.sleep(100)