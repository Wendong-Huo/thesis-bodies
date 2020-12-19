import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase

class Ant(WalkerBase):
  foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

  def __init__(self, xml):
    WalkerBase.__init__(self, xml, "torso", action_dim=8, obs_dim=28, power=2.5)

  def alive_bonus(self, z, pitch):
    return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

class AntBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, xml, render=False):
    self.robot = Ant(xml=xml)
    self.xml = xml
    WalkerBaseBulletEnv.__init__(self, self.robot, render)

  def show_body_id(self):
    if self._p:
      self._p.addUserDebugText(f"Body {self.xml[-5:]}", [-0.5,0,1], [0,0,1])