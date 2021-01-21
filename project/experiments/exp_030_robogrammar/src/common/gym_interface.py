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
    
def make_env(rank=0, seed=0, render=True, wrappers=[], robot_body=-1, body_info=-1, dataset_folder="../input_data/bodies", force_render=False):
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
                _render = rank in [0]
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



def make_pyrobotdesign_env(rank=0, seed=0, render=True, wrappers=[], robo_body="1000",  grammar_file = "grammar_apr30.dot", force_render=False, dataset_folder="../input_data/robo"):
    def _init():
        try:
            gym.spec("RoboEnv-v0")
        except:
            gym.envs.registration.register(id="RoboEnv-v0",
                entry_point="gym_envs.robo:RoboEnv",
                max_episode_steps=1000,
                reward_threshold=2500.0,
            )

        if force_render:
            _render = True
        else:
            _render = False
            if render:
                _render = rank in [0]
        with open(f"{dataset_folder}/{robo_body}.txt", "r") as f:
            rule_sequence = f.readline().strip()
        env = gym.make("RoboEnv-v0", grammar_file=grammar_file, rule_sequence=rule_sequence)
        env.robo_body = robo_body
        if _render:
            env.render()
        if len(wrappers)>0:
            for _wrapper in wrappers:
                env = _wrapper(env)
        if seed is not None:
            env.seed(seed*100 + rank)
            env.action_space.seed(seed*100 + rank)
        return env
    return _init
