import inspect
import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces


SPECIES = ("B", "C", "A")
_DEFAULT_INIT_POP = np.array([0.7, 0.6, 1.0], dtype=np.float32) #valores iniciales
_REWARD_TYPES = ("harvest_equilibrium", "equilibrium", "harvest") #los 3 tipos de recompensa
_EQ_CACHE = {} #guardar los equilibrios


def default_penalty(t):
    return -100.0 / (t + 1)


def _adapt_growth_fn(fn):  # para adaptar los numeros de parametros
    if fn is None:
        raise ValueError(
            "config['growth_fn'] es obligatorio, p.ej. growth_model_1_2 de "
            "foodweb_dynamics."
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


# recompensa

def estimate_equilibrium(growth_fn, parameters, initial_pop, n_runs=20,
                         n_steps=1000, burn_in=500, seed=0):
    rng = np.random.default_rng(seed)
    x0 = np.asarray(initial_pop, dtype=np.float64)
    n = x0.size
    run_means = []
    for _ in range(n_runs):
        pop = np.maximum(x0 * (1 + 0.05 * rng.standard_normal(n)), 0.0)
        total = np.zeros(n)
        for t in range(n_steps):
            pop = growth_fn(pop, parameters, t, rng)
            pop = np.maximum(np.asarray(pop, dtype=np.float64), 0.0)
            if t >= burn_in:
                total += pop
        run_means.append(total / (n_steps - burn_in))
    run_means = np.array(run_means)
    return run_means.mean(axis=0), run_means.std(axis=0)


def equilibrium_reward(pop, eq_pop, tol=1.0):
    pop = np.asarray(pop, dtype=np.float64)
    eq_pop = np.asarray(eq_pop, dtype=np.float64)
    d = np.linalg.norm(pop - eq_pop) / np.linalg.norm(eq_pop)
    return float(np.exp(-d / tol))


#entorno

class model_1_2(gym.Env):

    metadata = {"render_modes": []}
    num_species = 3
    species_names = SPECIES
    default_n_fished = 2

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = dict(config or {})
        cfg = self.config
        self.render_mode = render_mode

        if cfg.get("parameters") is None:
            raise ValueError("config['parameters'] es obligatorio, p.ej. parameters().parameters()")
        self.parameters = dict(cfg["parameters"])        # copia propia
        self.growth_fn = _adapt_growth_fn(cfg.get("growth_fn"))

        self.n_fished = int(cfg.get("n_fished", self.default_n_fished))
        if self.n_fished not in (1, 2):
            raise ValueError(f"n_fished debe ser 1 o 2, no {self.n_fished}")

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

        self.reward_type = cfg.get("reward_type", "harvest_equilibrium")
        if self.reward_type not in _REWARD_TYPES:
            raise ValueError(f"reward_type debe ser uno de {_REWARD_TYPES}")
        self.eq_tolerance = float(cfg.get("eq_tolerance", 1.0))
        self.eq_pop = None
        if self.reward_type != "harvest":
            self.eq_pop = self._get_equilibrium(cfg)

        self.action_space = spaces.Box(0.0, 1.0, shape=(self.n_fished,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)

        self.timestep = 0
        self.pop = self.initial_pop.copy()

        self.reset(seed=cfg.get("seed"))

    def _get_equilibrium(self, cfg):
        """x*: el de config['eq_pop'], o se estima promediando corridas."""
        if cfg.get("eq_pop") is not None:
            eq = np.asarray(cfg["eq_pop"], dtype=np.float64)
            if eq.size != self.num_species:
                raise ValueError(f"eq_pop tiene {eq.size} entradas; se esperaban 3.")
            return eq

        n_runs = int(cfg.get("eq_n_runs", 20))
        n_steps = int(cfg.get("eq_n_steps", 1000))
        burn_in = int(cfg.get("eq_burn_in", 500))
        seed = int(cfg.get("eq_seed", 0))

        fn_name = getattr(cfg["growth_fn"], "__qualname__", repr(cfg["growth_fn"]))
        key = (fn_name, repr(sorted(self.parameters.items())),
               self.initial_pop.tobytes(), n_runs, n_steps, burn_in, seed)
        if key not in _EQ_CACHE:
            eq, spread = estimate_equilibrium(
                self.growth_fn, self.parameters, self.initial_pop,
                n_runs, n_steps, burn_in, seed,
            )
            if np.any(spread > 0.05 * np.maximum(eq, 1e-8)):
                warnings.warn(
                    f"El equilibrio estimado varia entre corridas (x*={eq}, "
                    f"dispersion={spread}). Puede haber oscilaciones o mas de un "
                    f"equilibrio; revisar antes de confiar en la recompensa."
                )
            if np.any(eq <= self.threshold):
                warnings.warn(
                    f"El equilibrio estimado {eq} tiene una especie bajo el umbral "
                    f"de extincion ({self.threshold}): los episodios terminarian "
                    f"de inmediato."
                )
            _EQ_CACHE[key] = eq
        return _EQ_CACHE[key].copy()

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

        pop, harvest_reward = self.harvest(pop, action)
        pop = self.growth_fn(pop, self.parameters, self.timestep, self.np_random)
        pop = np.maximum(np.asarray(pop, dtype=np.float32), 0.0)   # sin densidades negativas

        # la recompensa se mide con la poblacion DESPUES del crecimiento
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
        fished = pop[: self.n_fished] * action
        pop[: self.n_fished] = np.maximum(pop[: self.n_fished] - fished, 0.0)
        reward = np.sum(fished * self.relative_weights - self.cost * action)
        return pop, float(reward)

    def _format_action(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size == 1 and self.n_fished > 1:
            a = np.repeat(a, self.n_fished)
        if a.size != self.n_fished:
            raise ValueError(
                f"La accion tiene {a.size} componentes; se esperaban {self.n_fished}."
            )
        return np.clip(a, 0.0, 1.0)

    def _get_obs(self):
        return np.clip(self.pop / self.bound, 0.0, 1.0).astype(np.float32)

    def _get_info(self):
        info = {
            "population": self.pop.copy(),
            "t": self.timestep,
            "species": self.species_names,
            "fished": self.species_names[: self.n_fished],
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



