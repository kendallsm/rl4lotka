"""Registra los ambientes"""

from gymnasium.envs.registration import register

from .model_1_2 import model_1_2
from .model_1_2_1 import model_1_2_1
from .model_1_3 import model_1_3
from .model_2_3 import model_2_3
from .params import config_for

ENVS = {
    "model_1_2": model_1_2,
    "model_1_3": model_1_3,
    "model_1_2_1": model_1_2_1,
    "model_2_3": model_2_3,
}


def _factory(name):
    env_cls = ENVS[name]

    def make(config=None, render_mode=None):
        config = dict(config or {})
        if "parameters" not in config:
            config = config_for(name, **config)
        return env_cls(config=config, render_mode=render_mode)

    return make


for _name in ENVS:
    register(id=_name, entry_point=_factory(_name))
