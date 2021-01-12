import pathlib
import gym
import common.common as common

def template(body):
    _template = {
        9: "walkerarms",
        8: "walkerarms",
        7: "walker2d",
        6: "hopper",
        5: "ant",
        4: "halfcheetah",
        3: "walker2d",
        0: "ant",
    }
    return _template[body//100]
    
def make_env(rank=0, seed=0, render=True, wrappers=[], robot_body=-1, body_info=-1, dataset_folder="../input_data/bodies"):
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
        
        _render = False
        if render:
            _render = rank in [0]
        env = gym.make(env_id, render=_render)
        env.rank = rank
        if len(wrappers)>0:
            for _wrapper in wrappers:
                env = _wrapper(env)
        if seed is not None:
            env.seed(seed*100 + rank)
            env.action_space.seed(seed*100 + rank)
        return env

    return _init
