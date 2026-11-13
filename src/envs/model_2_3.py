import inspect
import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces


SPECIES = ('A', 'B', 'C', 'D', 'E')
FISHED_IDX = (2, 4)                                  # C, E
PREDATION = ((0, 2), (0, 3), (1, 3), (1, 4))         # dinamica
COMPETITION = ()                                     # A y B compiten via D
_DEFAULT_INIT_POP = np.full(5, 0.5, dtype=np.float32) #valores para iniciar
_REWARD_TYPES = ("harvest_equilibrium", "equilibrium", "harvest") # los 3 tipos de recompensa


def default_penalty(t):
    return -100.0 / (t + 1)


def _adapt_growth_fn(fn):
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
        q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD) for q in sig.values()
    )
    if n_pos >= 3:
        return lambda pop, p, t, rng: fn(pop, p, t)
    return lambda pop, p, t, rng: fn(pop, p)


# la recompensa que deben tener dentro

def lv_equilibrium(r, A):
    #solución de la matriz Lotka-Volterra
    r = np.asarray(r, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    x_star = np.linalg.solve(A, -r)
    feasible = bool(np.all(x_star > 0))
    eig = np.linalg.eigvals(np.diag(x_star) @ A)
    stable = bool(np.all(eig.real < 0))
    return x_star, feasible, stable, eig


def equilibrium_reward(pop, eq_pop, tol=1.0):
 #cercanía al equilibrio
    pop = np.asarray(pop, dtype=np.float64)
    eq_pop = np.asarray(eq_pop, dtype=np.float64)
    d = np.linalg.norm(pop - eq_pop) / np.linalg.norm(eq_pop)
    return float(np.exp(-d / tol))


# entorno

class model_2_3(gym.Env):
    metadata = {"render_modes": []}
    num_species = 5
    species_names = SPECIES
    predation = PREDATION
    competition = COMPETITION

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = dict(config or {})
        cfg = self.config
        self.render_mode = render_mode

        if cfg.get("parameters") is None:
            raise ValueError(
            )
        self.parameters = dict(cfg["parameters"])          # copia propia
        self.growth_fn = _adapt_growth_fn(cfg.get("growth_fn"))

        self.fished_idx = np.atleast_1d(
            np.asarray(cfg.get("fished_idx", FISHED_IDX), dtype=int)
        )
        if self.fished_idx.min() < 0 or self.fished_idx.max() >= self.num_species:
            raise ValueError(f"fished_idx fuera de rango: {self.fished_idx.tolist()}")
        if len(set(self.fished_idx.tolist())) != self.fished_idx.size:
            raise ValueError("fished_idx tiene indices repetidos.")
        self.n_fished = int(self.fished_idx.size)

        self.initial_pop = np.asarray(
            cfg.get("initial_pop", _DEFAULT_INIT_POP), dtype=np.float32
        )
        if self.initial_pop.size != self.num_species:
            raise ValueError(
                f"initial_pop tiene {self.initial_pop.size} entradas; se esperaban "
                f"{self.num_species}."
            )

        self.Tmax = int(cfg.get("Tmax", 200))
        self.threshold = np.float32(cfg.get("threshold", 5e-2))
        self.init_sigma = np.float32(cfg.get("init_sigma", 5e-3))
        self.random_init = bool(cfg.get("random_init", False))

        self.cost = np.broadcast_to(
            np.asarray(cfg.get("cost", 0.0), dtype=np.float32), (self.n_fished,)
        ).copy()
        self.relative_weights = np.broadcast_to(
            np.asarray(cfg.get("relative_weights", 1.0), dtype=np.float32),
            (self.n_fished,),
        ).copy()

        self.bound = 2 * cfg.get("bound", 2)
        self.training = bool(cfg.get("training", True))
        self.early_end_penalty = cfg.get("early_end_penalty", default_penalty)

        self.reward_type = cfg.get("reward_type", "harvest_equilibrium")
        if self.reward_type not in _REWARD_TYPES:
            raise ValueError(f"reward_type debe ser uno de {_REWARD_TYPES}")
        self.eq_tolerance = float(cfg.get("eq_tolerance", 1.0))
        self.eq_pop = None
        if self.reward_type != "harvest":
            self.eq_pop = self._get_equilibrium(cfg)

        self.action_space = spaces.Box(0.0, 1.0, shape=(self.n_fished,), dtype=np.float32)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(self.num_species,), dtype=np.float32
        )

        self.timestep = 0
        self.pop = self.initial_pop.copy()

        self.reset(seed=cfg.get("seed"))

    def _get_equilibrium(self, cfg):
        if cfg.get("eq_pop") is not None:
            eq = np.asarray(cfg["eq_pop"], dtype=np.float64)
            if eq.size != self.num_species:
                raise ValueError(
                    f"eq_pop tiene {eq.size} entradas; se esperaban {self.num_species}."
                )
            return eq

        if "r" not in self.parameters or "A" not in self.parameters:
            raise ValueError(
            )
        eq, feasible, stable, eig = lv_equilibrium(self.parameters["r"], self.parameters["A"])
        if not feasible:
            raise ValueError(
            )
        if not stable:
            warnings.warn("equilibrio inestable")
        if np.any(eq <= self.threshold):
            warnings.warn("equilibrio bajo el umbral de extincion")
        return eq

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.timestep = 0
        n = self.num_species
        if "initial_pop" in options:
            pop = np.asarray(options["initial_pop"], dtype=np.float32)
        elif self.random_init:
            pop = self.np_random.random(n)
        else:
            pop = self.initial_pop + self.bound * self.init_sigma * self.np_random.normal(size=n)
        self.pop = np.maximum(pop, 0.0).astype(np.float32)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = self._format_action(action)
        pop = self.pop.copy()

        pop, harvest_reward = self.harvest(pop, action)
        pop = self.growth_fn(pop, self.parameters, self.timestep, self.np_random)
        pop = np.maximum(np.asarray(pop, dtype=np.float32), 0.0)

        # la recompensa se mide con la poblacion con el crecimiento
        if self.reward_type == "harvest_equilibrium":
            reward = harvest_reward * equilibrium_reward(pop, self.eq_pop, self.eq_tolerance)
        elif self.reward_type == "equilibrium":
            reward = equilibrium_reward(pop, self.eq_pop, self.eq_tolerance)
        else:
            reward = harvest_reward

        terminated = bool(self.training and np.any(pop <= self.threshold))
        if terminated:
            reward += self.early_end_penalty(self.timestep)

        self.pop = pop
        self.timestep += 1
        truncated = (not terminated) and self.timestep >= self.Tmax
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def harvest(self, pop, action):
        idx = self.fished_idx
        fished = pop[idx] * action
        pop[idx] = np.maximum(pop[idx] - fished, 0.0)
        reward = np.sum(fished * self.relative_weights - self.cost * action)
        return pop, float(reward)

    def _format_action(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size == 1 and self.n_fished > 1:        # esfuerzo unico compartido
            a = np.repeat(a, self.n_fished)
        if a.size != self.n_fished:
            raise ValueError(
            )
        return np.clip(a, 0.0, 1.0)

    def _get_obs(self):
        return np.clip(self.pop / self.bound, 0.0, 1.0).astype(np.float32)

    def _get_info(self):
        info = {
            "population": self.pop.copy(),
            "t": self.timestep,
            "species": self.species_names,
            "fished": tuple(self.species_names[i] for i in self.fished_idx),
        }
        if self.eq_pop is not None:
            info["eq_distance"] = float(
                np.linalg.norm(self.pop - self.eq_pop) / np.linalg.norm(self.eq_pop)
            )
        return info

    def update_state(self, pop):
        return np.asarray(pop, dtype=np.float32) / self.bound

    def population(self):
        return self.pop.copy()

    @property
    def state(self):
        return self._get_obs()
