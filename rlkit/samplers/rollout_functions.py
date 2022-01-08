import numpy as np
from rlkit.samplers.rollout import Rollout


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {key: flatten_n([d[key] for d in dicts]) for key in keys}


def vec_multitask_rollout(
    env,
    agent,
    envs_rollout,
    obs_reset,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    representation_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    reset_kwargs=None,
):

    return None


def multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    representation_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    reset_kwargs=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = {}
    next_observations = []
    path_length = 0
    if reset_kwargs:
        o = env.reset(**reset_kwargs)
    else:
        o = env.reset()
    agent.reset()
    if render:
        env.render(**render_kwargs)
    desired_goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            s = o[observation_key]
        g = o[representation_goal_key]
        new_obs = np.hstack((s, g))
        if agent.spirl == False:
            a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
            next_o, r, d, env_info = env.step(a)
        else:
            a,z,agent_info = agent.get_action(new_obs,**get_action_kwargs)
            r = np.zeros(1)
            for act in a:
                print(act)
                next_o, R, d, env_info = env.step(act)
                r= r + R
                if d:
                    break
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        if agent.spirl ==False:
            actions.append(a)
        else:
            actions.append(z)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        if not env_infos:
            for k, v in env_info.items():
                env_infos[k] = [v]
        else:
            for k, v in env_info.items():
                env_infos[k].append(v)
        path_length += 1
        if d:
            break
        o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    for k, v in env_infos.items():
        env_infos[k] = np.array(v)
    return dict(
        observations=observations,
        actions=actions,
        # rewards=np.array(rewards).reshape(-1, 1),
        rewards=np.array(rewards),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        desired_goals=np.repeat(desired_goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def multiagent_multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    achieved_q_key=None,
    desired_q_key=None,
    representation_goal_key=None,
    get_action_kwargs=None,
    reset_kwargs=None,
):

    return None


def plot_paths(paths):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for path, color in zip(paths, ["green", "red"]):
        print(path["terminals"][-1], path["rewards"][-1])
        path = [path["observations"][0]] + list(path["next_observations"])
        for i in range(len(path) - 1):
            p = path[i]["achieved_goal"]
            next_p = path[i + 1]["achieved_goal"]
            ax.scatter(p[0], p[1], c=color)
            ax.scatter(next_p[0], next_p[1], c=color)
            ax.plot([p[0], next_p[0]], [p[1], next_p[1]], c=color)
    plt.show()


def rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    reset_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = {}
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if reset_kwargs:
        o = env.reset(**reset_kwargs)
    else:
        o = env.reset()
    if render:
        env.render(**render_kwargs)

    while path_length < max_path_length:
        if agent.spirl == False:
            a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
            next_o, r, d, env_info = env.step(a)
        else:
            a,z,agent_info = agent.get_action(new_obs,**get_action_kwargs)
            r = np.zeros(1)
            for act in a:
                next_o, R, d, env_info = env.step(act)
                r = r + R
                if d:
                    break
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        if agent.spirl ==False:
            actions.append(a)
        else:
            actions.append(z)
        agent_infos.append(agent_info)
        if not env_infos:
            for k, v in env_info.items():
                env_infos[k] = [v]
        else:
            for k, v in env_info.items():
                env_infos[k].append(v)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    for k, v in env_infos.items():
        env_infos[k] = np.array(v)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
