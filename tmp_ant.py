import pybullet as p
from PyBulletWrapper.pybullet_wrapper.handy import HandyPyBullet

p = HandyPyBullet(p)
p.connectPy()
# p.loadMJCF("cleaned_version/body-templates/ant.original.xml")
p.loadMJCF("cleaned_version/dataset/ant_20_10-v0/bodies/1.xml")

while True:
    p.stepSimulation()
    p.sleepPy()