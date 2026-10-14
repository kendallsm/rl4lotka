import inspect

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_DEFAULT_INIT_POP = np.array([0.7, 0.6, 1.0], dtype=np.float32)


def default_penalty(t):
    return -100.0 / (t + 1)


def _adapt_growth_fn(fn): #para adaptar los numeros de parametros 
    if fn is None:
        raise ValueError(
        )
    try:
        sig = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn
    if "rng" in sig:
        return fn
    n_pos = sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in sig.values()
    )
    if n_pos >= 3:
        return lambda pop, p, t, rng: fn(pop, p, t)
    return lambda pop, p, t, rng: fn(pop, p)


class _ThreeSpeciesFishing(gym.Env):

    metadata = {"render_modes": []}
    n_fished = 2

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        # self.config = config or {}  ahora  config.get
        self.config = dict(config or {})
        cfg = self.config
        self.render_mode = render_mode

        if cfg.get("parameters") is None:
            raise ValueError("config['parameters'] es obligatorio, p.ej. parameters().parameters()")
        self.parameters = dict(cfg["parameters"])        # copia propia
        self.growth_fn = _adapt_growth_fn(cfg.get("growth_fn"))
        self.num_species = 3

        self.initial_pop = np.asarray(cfg.get("initial_pop", _DEFAULT_INIT_POP), dtype=np.float32)
        self.Tmax = int(cfg.get("Tmax", 200))
        self.threshold = np.float32(cfg.get("threshold", 5e-2))
        self.init_sigma = np.float32(cfg.get("init_sigma", 5e-3))
        self.random_init = bool(cfg.get("random_init", False))

        self.cost = np.broadcast_to(
            np.asarray(cfg.get("cost", 0.0), dtype=np.float32), (self.n_fished,)
        ).copy()
        self.relative_weights = np.broadcast_to(
            np.asarray(cfg.get("relative_weights", 1.0), dtype=np.float32), (self.n_fished,)
        ).copy()
        self.bound = 2 * cfg.get("bound", 2)
        self.training = bool(cfg.get("training", True))
        self.early_end_penalty = cfg.get("early_end_penalty", default_penalty)
        

        self.action_space = spaces.Box(0.0, 1.0, shape=(self.n_fished,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)

        self.timestep = 0
        self.pop = self.initial_pop.copy()
        
        self.reset(seed=cfg.get("seed"))

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)              
        options = options or {}
        self.timestep = 0
        if "initial_pop" in options:          
            pop = np.asarray(options["initial_pop"], dtype=np.float32)
        elif self.random_init:
            pop = self.np_random.random(3)
        else:
            
            pop = self.initial_pop + self.bound * self.init_sigma * self.np_random.normal(size=3)
        self.pop = np.maximum(pop, 0.0).astype(np.float32)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = self._format_action(action)
        pop = self.pop.copy()

        pop, reward = self.harvest(pop, action)
        # antes: if fluctuating: growth_fn(pop, p, t) else growth_fn(pop, p)
        pop = self.growth_fn(pop, self.parameters, self.timestep, self.np_random)
        pop = np.maximum(np.asarray(pop, dtype=np.float32), 0.0)   # sin densidades negativas

        terminated = bool(self.training and np.any(pop <= self.threshold))
        if terminated:
            reward += self.early_end_penalty(self.timestep)

        self.pop = pop
        self.timestep += 1
        # antes: siempre False
        truncated = (not terminated) and self.timestep >= self.Tmax
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    
    def harvest(self, pop, action):
        fished = pop[: self.n_fished] * action
        pop[: self.n_fished] = np.maximum(pop[: self.n_fished] - fished, 0.0)
        reward = np.sum(fished * self.relative_weights - self.cost * action)
        return pop, float(reward)

    def _format_action(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size == 1 and self.n_fished > 1:   
            a = np.repeat(a, self.n_fished)
        return np.clip(a, 0.0, 1.0)

    def _get_obs(self):
        return np.clip(self.pop / self.bound, 0.0, 1.0).astype(np.float32)

    def _get_info(self):
        return {"population": self.pop.copy(), "t": self.timestep}

   
    def update_state(self, pop):
        return np.asarray(pop, dtype=np.float32) / self.bound

    def population(self):
        return self.pop.copy()

    @property
    def state(self):
        return self._get_obs()


class twoThreeFishing(_ThreeSpeciesFishing):
    """3 especies, se pescan X e Y."""
    n_fished = 2


class oneThreeFishing(_ThreeSpeciesFishing):
    """3 especies, se pesca solo X."""
    n_fished = 1

