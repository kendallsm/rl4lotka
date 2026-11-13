"""rl4lotka: aprendizaje por refuerzo en redes troficas.

    from rl4lotka import train
    save_id, options = train({"algo": "PPO", "model": "model_1_3", "total_timesteps": 100_000})
"""

from .train.sb3 import load_agent, sb3_train, train

__all__ = ["train", "sb3_train", "load_agent"]
