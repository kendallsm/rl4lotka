import numpy as np


class FixedEffort:
    """Politica fija: el mismo esfuerzo de pesca en todos los pasos (0 = sin pesca)."""

    def __init__(self, env, effort=0.0):
        self.effort = float(effort)
        self.n = int(env.action_space.shape[0])

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return np.full(self.n, self.effort, dtype=np.float32), None


def _es_recurrente(agent):
    policy = getattr(agent, "policy", None)
    return hasattr(policy, "lstm_actor")


def run_episode(env, agent):
    """Un episodio. Devuelve dict con rew, length y eq_distance final."""
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    recurrente = _es_recurrente(agent)
    state, start = None, np.array([True])

    obs, info = env.reset()
    episode_reward = 0.0
    length = 0
    for t in range(env.Tmax):
        if recurrente:
            action, state = agent.predict(
                obs, state=state, episode_start=start, deterministic=True
            )
            start = np.array([False])
        else:
            action = agent.predict(obs, deterministic=True)[0]
        obs, reward, terminated, done, info = env.step(action)
        episode_reward += reward
        length += 1
        if terminated or done:
            break
    return {
        "rew": float(episode_reward),
        "length": length,
        "eq_distance": info.get("eq_distance", np.nan),
    }


def get_simulator(ray_remote=False):
    if ray_remote:
        import ray

        ray.init(logging_level="error")
        return ray.remote(run_episode)
    return run_episode


class evaluate_agent:
    def __init__(self, *, agent, env=None, ray_remote=False):
        self.agent = agent
        self.env = env or agent.env
        self.simulator = get_simulator(ray_remote=ray_remote)
        self.ray_remote = ray_remote

    def episodes(self, n_eval_episodes=50):
        """Lista de dicts (rew, length, eq_distance), uno por episodio."""
        if self.ray_remote:
            import ray

            out = ray.get([
                self.simulator.remote(self.env, self.agent) for _ in range(n_eval_episodes)
            ])
            if ray.is_initialized():
                ray.shutdown()
            return out
        return [self.simulator(self.env, self.agent) for _ in range(n_eval_episodes)]

    def evaluate(self, return_episode_rewards=False, n_eval_episodes=50):
        rewards = [e["rew"] for e in self.episodes(n_eval_episodes)]
        return rewards if return_episode_rewards else np.mean(rewards)
