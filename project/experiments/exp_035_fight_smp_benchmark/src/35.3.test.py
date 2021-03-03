# copy xmls from smp repo
# replace all `name="root` into `name="ignore` to make it readable to pybullet
# delete floor

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
import time
import pathlib
import gym
import pybullet as p
import cv2
from common import common
common.args.custom_align_max_joints = 6
common.args.num_venvs = 12
args = common.args
saved_model_filename = f"sd{args.seed}"
from common import gym_interface, wrapper_custom_align, callbacks

def make_env(robot_body, rank=0, seed=0, visualize_rank=0):
    def _init():
        env_id = f"{robot_body}-v0"
        xml_path = f"../input_data/smp_envs/xmls/{robot_body}.xml"
        xml_path = str(pathlib.Path(xml_path).resolve())

        def get_number_joints(xml_path):
            p.connect(p.DIRECT)
            (robot,) = p.loadMJCF(xml_path)
            print("")
            num_joints = p.getNumJoints(robot)
            num_real_joints = 0
            for i in range(num_joints):
                info = p.getJointInfo(robot, i)
                joint_name = info[1].decode()
                if "ignore" in joint_name or "root" in joint_name or "fix_" in joint_name:
                    continue
                num_real_joints += 1
                print(num_real_joints, joint_name)
            p.disconnect()
            return num_real_joints
        num_real_joints = get_number_joints(xml_path)

        print(f"number of joints: {num_real_joints}")
        try:
            gym.spec(env_id)
        except:
            gym.envs.registration.register(id=env_id,
                                           entry_point=f'gym_envs.randomrobot:MyRandomRobotEnv',
                                           max_episode_steps=1000,
                                           reward_threshold=2500.0,
                                           kwargs={"xml": xml_path, "num_joints": num_real_joints, "self_collision": False, "power": 0.9})

        env = gym.make(env_id, render=(rank == visualize_rank))
        # rank related to which order is used for each body, so rank and the test_bodies need to be in the same order.
        env.rank = rank
        if seed is not None:
            env.seed(seed*100 + rank)
            env.action_space.seed(seed*100 + rank)
        env = wrapper_custom_align.CustomAlignWrapper(env, max_num_joints=6)
        return env
    return _init


# cheetah_2_back cheetah_3_balanced
# cheetah_2_front cheetah_5_back
# cheetah_3_back cheetah_6_front
# cheetah_3_front
# cheetah_4_allback
# cheetah_4_allfront
# cheetah_4_back
# cheetah_4_front
# cheetah_5_balanced
# cheetah_5_front
# cheetah_6_back
# cheetah_7_full
robot_bodies = ["cheetah_2_back", "cheetah_2_front", "cheetah_3_back", "cheetah_3_front", "cheetah_4_allback", "cheetah_4_allfront",
                "cheetah_4_back", "cheetah_4_front", "cheetah_5_balanced", "cheetah_5_front", "cheetah_6_back", "cheetah_7_full"]
zero_shot_bodies = ["cheetah_3_balanced", "cheetah_5_back", "cheetah_6_front"]
cheetah_orders = {
    "cheetah_2_back": "0,1,2,3,4,5",
    "cheetah_2_front": "3,1,2,0,4,5",
    "cheetah_3_back": "0,1,2,3,4,5",
    "cheetah_3_front": "3,4,0,1,2,5",
    "cheetah_4_allback": "0,1,2,3,4,5",
    "cheetah_4_allfront": "3,4,5,0,1,2",
    "cheetah_4_back": "0,1,3,2,4,5",
    "cheetah_4_front": "0,3,4,1,2,5",
    "cheetah_5_balanced": "0,1,3,4,2,5",
    "cheetah_5_front": "0,3,4,5,1,2",
    "cheetah_6_back": "0,1,2,3,4,5",
    "cheetah_7_full": "0,1,2,3,4,5",

    "cheetah_3_balanced": "0,3,1,2,4,5",
    "cheetah_5_back": "0,1,2,3,4,5",
    "cheetah_6_front": "0,1,3,4,5,2",
}


def take_a_photo():
    for robot_body in robot_bodies+zero_shot_bodies:
        if args.smp_bodies_aligned:
            common.args.custom_alignment = cheetah_orders[robot_body]
        env = make_env(robot_body)()
        env.reset()
        _p = env.env.env._p
        _p.resetDebugVisualizerCamera(2, 0, -10, [0, 0, 0.2])
        common.linux_fullscreen()
        time.sleep(1)
        (width, height, rgbPixels, _, _) = _p.getCameraImage(1920, 1080, renderer=_p.ER_BULLET_HARDWARE_OPENGL)
        image = rgbPixels[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print("Save to ", f"output_data/saved_images/{robot_body}.png")
        print(image.shape)
        cv2.imwrite(f"output_data/saved_images/{robot_body}_original.png", image)
        env.close()

# time.sleep(100)

if args.smp_bodies_aligned:
    common.args.custom_alignment = "::".join([cheetah_orders[x] for x in robot_bodies])
hyperparams = common.load_hyperparameters(conf_name=args.rl_hyperparameter)

common.args.model_filename = "output_data/tmp/aligned/sd0/best_model.zip"

# for test_body in robot_bodies+zero_shot_bodies:
for test_body in ["cheetah_3_balanced"]:
    if args.smp_bodies_aligned:
        common.args.custom_alignment = cheetah_orders[test_body]
    # if test_body == "cheetah_6_front":
    #     eval_venv = DummyVecEnv([make_env(test_body)])
    # else:
    eval_venv = DummyVecEnv([make_env(test_body, rank=0)]) # 1 avoid visualization

    if args.vec_normalize:
        eval_venv = VecNormalize.load(common.get_vec_pkl_from_model_filename(args.model_filename), eval_venv)

    hyperparams = common.clean_hyperparams_before_run(hyperparams)
    model = PPO.load(common.args.model_filename, env=eval_venv, **hyperparams)

    obs = eval_venv.reset()

    while True:
        action = model.predict(obs)
        obs, reward, done, _ = eval_venv.step(action[0])
        if done:
            break


    eval_venv.close()
