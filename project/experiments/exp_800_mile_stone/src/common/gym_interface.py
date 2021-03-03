import pathlib
import gym
import common.common as common

def template(body):
    _template = {
        9: "walkerarms",
        8: "randomrobot",
        7: "walker2d",
        6: "hopper",
        5: "ant",
        4: "halfcheetah",
        3: "walker2d",
        2: "randomrobot",
        1: "randomrobot",
        0: "ant",
    }
    return _template[body//100]

def make_env(rank=0, seed=0, render=True, wrappers=[], robot_body=-1, body_info=-1, dataset_folder="../input_data/bodies", force_render=False, render_index=0):
    # print(f"make_env( rank={rank}, seed={seed}, wrapper={'None' if wrapper is None else wrapper.__name__}, robot_body={robot_body}, body_info={body_info}")
    _template=template(robot_body)
    if dataset_folder is None:
        dataset_folder = f"{str(common.input_data_folder.resolve())}/bodies"
    else:
        if isinstance(dataset_folder,str):
            dataset_folder = str(pathlib.Path(dataset_folder).resolve())
        else:
            print("Warning: not a string.")
            dataset_folder = str(dataset_folder)

    def _init():
        env_names = {
            "ant": "MyAntBulletEnv",
            "walker2d": "MyWalker2DBulletEnv",
            "hopper": "MyHopperBulletEnv",
            "halfcheetah": "MyHalfCheetahBulletEnv",
            "walkertail": "MyWalkerTailBulletEnv",
            "walkerarms": "MyWalkerArmsBulletEnv",
            "randomrobot": "MyRandomRobotEnv",
        }
        env_name = env_names[_template]
        env_id = f'{env_name}-v{robot_body}'
        try:
            gym.spec(env_id)
        except:
            gym.envs.registration.register(id=env_id,
                entry_point=f'gym_envs.{_template}:{env_name}',
                max_episode_steps=1000,
                reward_threshold=2500.0, 
                kwargs={"xml":f"{dataset_folder}/{robot_body}.xml"})
        
        if force_render:
            _render = True
        else:
            _render = False
            if render:
                _render = rank in [render_index]
        env = gym.make(env_id, render=_render)
        # rank related to which order is used for each body, so rank and the test_bodies need to be in the same order.
        env.rank = rank
        if len(wrappers)>0:
            for _wrapper in wrappers:
                env = _wrapper(env)
        if seed is not None:
            env.seed(seed*100 + rank)
            env.action_space.seed(seed*100 + rank)
        return env

    return _init

def get_max_num_joints(bodies=[300]):
    max_num_joints = -1
    for body in bodies:
        _env = make_env(robot_body=body, render=False)
        _env = _env()
        _env.reset()
        num_joints = len(_env.robot.jdict)
        # print(num_joints)
        if max_num_joints< num_joints:
            max_num_joints = num_joints
        _env.close()
    return max_num_joints