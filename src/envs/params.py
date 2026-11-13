"""
Para cambiar algunos de los parámetros, se pone en options["config"], por ejemplo si se cambia el sd
{"s_D": 0.1, "Tmax": 500}.

Orden de los vectores:

    model_1_2   : [B, C, A]        las pescadas primero: el entorno de 3
                                   especies cosecha pop[:n_fished]
    model_1_3   : [A, B, C, D]
    model_1_2_1 : [A, B, C, D]
    model_2_3   : [A, B, C, D, E]

Equilibrio de referencia (el punto contra el que se medimos la recompensa):

    model_1_2              se estima simulando sin pesca y promediando
                           corridas (estimate_equilibrium)
    model_1_3, model_1_2_1,
    model_2_3              Lotka-Volterra generalizado, dx/dt = x (r + A x);
                           el equilibrio interior resuelve A x* = -r
                           (lv_equilibrium).
"""

import numpy as np

__all__ = [
    "parameters_model_1_2", "growth_model_1_2",
    "parameters_model_1_3", "growth_model_1_3",
    "parameters_model_1_2_1", "growth_model_1_2_1",
    "parameters_model_2_3", "growth_model_2_3",
    "growth_lv", "lv_equilibrium", "estimate_equilibrium",
    "MODELS", "config_for",
]


# infraestructura

def _copy(v):
    return v.copy() if isinstance(v, np.ndarray) else v


class _Parameters:

    defaults = {}

    def __init__(self, **overrides):
        desconocidos = set(overrides) - set(self.defaults)
        if desconocidos:
            raise KeyError(
                f"parametros desconocidos para {type(self).__name__}: "
                f"{sorted(desconocidos)}"
            )
        self._p = {k: _copy(v) for k, v in self.defaults.items()}
        self._p.update(overrides)

    def raw(self):
        return {k: _copy(v) for k, v in self._p.items()}

    def parameters(self):
        """#recibe la funcion"""
        return self.raw()

    def equilibrium(self):
        raise NotImplementedError

    def init_state(self):
        """estado inicial"""
        return np.asarray(self.equilibrium(), dtype=np.float32)

    def __repr__(self):
        return f"{type(self).__name__}({self._p})"


def _integrate(pop, p, rhs, rng):
    x = np.asarray(pop, dtype=np.float64)
    n = max(int(p.get("substeps", 20)), 1)
    h = p.get("dt", 1.0) / n
    for _ in range(n):
        x = np.maximum(x + h * rhs(x, p), 0.0)

    sigma = p.get("sigma", 0.0)
    if sigma > 0.0:
        x = x * np.exp(sigma * rng.normal(size=x.size) - 0.5 * sigma ** 2)
    return np.maximum(x, 0.0)


# ======================================================================
# Equilibrios
# ======================================================================

def lv_equilibrium(r, A):
    """equilibrio Lotka-Volterra (lo calculamos afuera provisionalmente)
    """
    r = np.asarray(r, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    x_star = np.linalg.solve(A, -r)
    feasible = bool(np.all(x_star > 0))
    eig = np.linalg.eigvals(np.diag(x_star) @ A)
    stable = bool(np.all(eig.real < 0))
    return x_star, feasible, stable, eig


def estimate_equilibrium(growth_fn, parameters, initial_pop, n_runs=20,
                         n_steps=1000, burn_in=500, seed=0):
    """Estima el equilibrio simulando sin pesca y promediando corridas.
    """
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


#Lotka-Volterra

def _rhs_lv(x, p):
    return x * (p["r"] + p["A"] @ x)


def growth_lv(pop, p, t, rng):
    """Un paso de dx/dt = x (r + A x). Sirve para las tres redes LV."""
    return _integrate(pop, p, _rhs_lv, rng)


class _LVParameters(_Parameters):
    """Base de las redes LV: los coeficientes ecologicos arman r y A.

    Notacion de los coeficientes (todos >= 0; los signos los pone build):
        r_i   crecimiento intrinseco de una presa basal
        m_i   mortalidad de un depredador (entra como r_i = -m_i)
        s_i   autolimitacion (A_ii = -s_i). En presas basales s_i = r_i / K_i
        c_ij  competencia directa entre i y j (A_ij = A_ji = -c_ij)
        a_ij  tasa de ataque del depredador i sobre la presa j (A_ji = -a_ij)
        e_i   eficiencia de conversion del depredador i (A_ij = e_i a_ij)
    """

    def build(self):
        """Devuelve (r, A) a partir de los coeficientes que metamos."""
        raise NotImplementedError

    def parameters(self):
        r, A = self.build()
        p = self.raw()
        return {
            "r": r,
            "A": A,
            "dt": p["dt"],
            "substeps": p["substeps"],
            "sigma": p["sigma"],
        }

    def equilibrium(self):
        r, A = self.build()
        x_star, feasible, stable, _ = lv_equilibrium(r, A)
        if not feasible:
            raise ValueError(
                f"{type(self).__name__}: el equilibrio interior {x_star} no es "
                f"factible (alguna especie <= 0). Con estos parametros no hay "
                f"coexistencia."
            )
        return x_star

    def stability(self):
        r, A = self.build()
        return lv_equilibrium(r, A)


# Parametros para cada modelo

class parameters_model_1_3(_LVParameters):

    defaults = {
        "r_A": 1.0, "r_B": 1.0, "r_C": 1.0,        # crecimiento de las presas
        "s_A": 1.0, "s_B": 1.0, "s_C": 1.0,        # autolimitacion de presas
        "c_AB": 0.4, "c_BC": 0.4,                  # competencia (A y C no compiten)
        "a_DA": 0.5, "a_DB": 0.5, "a_DC": 0.5,     # ataque de D
        "e_D": 0.4,                                # conversion de D
        "m_D": 0.2, "s_D": 0.0,                    # mortalidad y autolimitacion de D
        "dt": 1.0, "substeps": 20, "sigma": 0.02,
    }

    def build(self):
        p = self._p
        r = np.array([p["r_A"], p["r_B"], p["r_C"], -p["m_D"]])
        A = np.array([
            [-p["s_A"],  -p["c_AB"],  0.0,        -p["a_DA"]],
            [-p["c_AB"], -p["s_B"],   -p["c_BC"], -p["a_DB"]],
            [0.0,        -p["c_BC"],  -p["s_C"],  -p["a_DC"]],
            [p["e_D"] * p["a_DA"], p["e_D"] * p["a_DB"], p["e_D"] * p["a_DC"], -p["s_D"]],
        ])
        return r, A


growth_model_1_3 = growth_lv


class parameters_model_1_2_1(_LVParameters):
    defaults = {
        "m_A": 0.1, "s_A": 0.2, "a_AB": 0.6, "a_AC": 0.6, "e_A": 0.5,
        "m_B": 0.1, "s_B": 0.15, "a_BD": 1.2, "e_B": 0.5,
        "m_C": 0.1, "s_C": 0.15, "a_CD": 1.2, "e_C": 0.5,
        "r_D": 1.0, "s_D": 1.0,
        "dt": 1.0, "substeps": 20, "sigma": 0.02,
    }

    def build(self):
        p = self._p
        r = np.array([-p["m_A"], -p["m_B"], -p["m_C"], p["r_D"]])
        A = np.array([
            [-p["s_A"], p["e_A"] * p["a_AB"], p["e_A"] * p["a_AC"], 0.0],
            [-p["a_AB"], -p["s_B"], 0.0, p["e_B"] * p["a_BD"]],
            [-p["a_AC"], 0.0, -p["s_C"], p["e_C"] * p["a_CD"]],
            [0.0, -p["a_BD"], -p["a_CD"], -p["s_D"]],
        ])
        return r, A


growth_model_1_2_1 = growth_lv

class parameters_model_2_3(_LVParameters):
    defaults = {
        "m_A": 0.2, "s_A": 0.1, "a_AC": 0.6, "a_AD": 0.6, "e_A": 0.4,
        "m_B": 0.2, "s_B": 0.1, "a_BD": 0.6, "a_BE": 0.6, "e_B": 0.4,
        "r_C": 1.0, "s_C": 1.0,
        "r_D": 1.1, "s_D": 1.1,
        "r_E": 0.9, "s_E": 0.9,
        "dt": 1.0, "substeps": 20, "sigma": 0.02,
    }

    def build(self):
        p = self._p
        r = np.array([-p["m_A"], -p["m_B"], p["r_C"], p["r_D"], p["r_E"]])
        A = np.array([
            [-p["s_A"], 0.0, p["e_A"] * p["a_AC"], p["e_A"] * p["a_AD"], 0.0],
            [0.0, -p["s_B"], 0.0, p["e_B"] * p["a_BD"], p["e_B"] * p["a_BE"]],
            [-p["a_AC"], 0.0, -p["s_C"], 0.0, 0.0],
            [-p["a_AD"], -p["a_BD"], 0.0, -p["s_D"], 0.0],
            [0.0, -p["a_BE"], 0.0, 0.0, -p["s_E"]],
        ])
        return r, A


growth_model_2_3 = growth_lv

# ======================================================================
# model_1_2   A -> B, C ;  B -- C        estado [B, C, A]
# ======================================================================

_EQ_CACHE_1_2 = {}


class parameters_model_1_2(_Parameters):
    defaults = {
        "r_x": 1.0, "r_y": 1.0,      # crecimiento de las presas B y C
        "K_x": 1.0, "K_y": 1.0,      # capacidades de carga
        "beta": 0.3,                 # ataque maximo del depredador A
        "v0": 0.3,                   # respuesta
        "D": 1.1,                    # cuanto mas ataca A a C que a B
        "tau_yx": 0.0, "tau_xy": 0.0,  # migracion entre B e C (apagada)
        "cV": 0.1,                   # competencia B-C
        "f": 0.1,                    # conversion de presa en depredador
        "dH": 0.1,                   # mortalidad del depredador
        "alpha": 1.0,                # escala temporal del depredador
        "sigma_x": 0.1, "sigma_y": 0.1, "sigma_z": 0.1,   # ruido
    }

    # punto de partida de las corridas del promedio(paper)
    _start = (0.5, 0.5, 0.7)

    def equilibrium(self, n_runs=20, n_steps=1000, burn_in=500, seed=0):
        key = (repr(sorted(self._p.items())), n_runs, n_steps, burn_in, seed)
        if key not in _EQ_CACHE_1_2:
            eq, _ = estimate_equilibrium(
                growth_model_1_2, self.parameters(), self._start,
                n_runs, n_steps, burn_in, seed,
            )
            _EQ_CACHE_1_2[key] = eq
        return _EQ_CACHE_1_2[key].copy()


def growth_model_1_2(pop, p, t, rng):
    X, Y, Z = (float(v) for v in pop)
    c = p["v0"] ** 2
    nX = X + (p["r_x"] * X * (1 - X / p["K_x"])
              - p["beta"] * Z * X ** 2 / (c + X ** 2)
              - p["cV"] * X * Y
              + p["tau_yx"] * Y - p["tau_xy"] * X
              + p["sigma_x"] * X * rng.normal())
    nY = Y + (p["r_y"] * Y * (1 - Y / p["K_y"])
              - p["D"] * p["beta"] * Z * Y ** 2 / (c + Y ** 2)
              - p["cV"] * X * Y
              - p["tau_yx"] * Y + p["tau_xy"] * X
              + p["sigma_y"] * Y * rng.normal())
    nZ = Z + p["alpha"] * (Z * (p["f"] * (X + p["D"] * Y) - p["dH"])
                           + p["sigma_z"] * Z * rng.normal())
    return np.maximum(np.array([nX, nY, nZ]), 0.0)


# registramos modelos

MODELS = {
    "model_1_2": {
        "figura": "a / b",
        "parameters": parameters_model_1_2,
        "growth_fn": growth_model_1_2,
        "especies": ("B", "C", "A"),
        "fished_idx": (0, 1),           # n_fished = 2; con n_fished = 1 solo B
        "equilibrio": "promedio de corridas",
    },
    "model_1_3": {
        "figura": "c",
        "parameters": parameters_model_1_3,
        "growth_fn": growth_model_1_3,
        "especies": ("A", "B", "C", "D"),
        "fished_idx": (0, 1, 2),
        "equilibrio": "Lotka-Volterra, A x* = -r",
    },
    "model_1_2_1": {
        "figura": "d",
        "parameters": parameters_model_1_2_1,
        "growth_fn": growth_model_1_2_1,
        "especies": ("A", "B", "C", "D"),
        "fished_idx": (1, 2),
        "equilibrio": "Lotka-Volterra, A x* = -r",
    },
    "model_2_3": {
        "figura": "e",
        "parameters": parameters_model_2_3,
        "growth_fn": growth_model_2_3,
        "especies": ("A", "B", "C", "D", "E"),
        "fished_idx": (2, 4),
        "equilibrio": "Lotka-Volterra, A x* = -r",
    },
}

# Opciones que entienden los entornos para evitar typos
_ENV_OPTIONS = frozenset({
    "parameters", "growth_fn", "initial_pop", "fished_idx", "n_fished",
    "Tmax", "threshold", "init_sigma", "random_init", "cost",
    "relative_weights", "bound", "training", "early_end_penalty", "seed",
    "reward_type", "eq_pop", "eq_tolerance",
    "eq_n_runs", "eq_n_steps", "eq_burn_in", "eq_seed",
})


def config_for(model, **overrides):
    """Arma el config"""
    if model not in MODELS:
        raise KeyError(f"modelo desconocido: {model!r}. Opciones: {sorted(MODELS)}")

    spec = MODELS[model]
    cls = spec["parameters"]

    de_dinamica = {k: v for k, v in overrides.items() if k in cls.defaults}
    de_entorno = {k: v for k, v in overrides.items()
                  if k not in cls.defaults and k in _ENV_OPTIONS}
    desconocidos = set(overrides) - set(de_dinamica) - set(de_entorno)
    if desconocidos:
        raise KeyError(
            f"nombres desconocidos para {model!r}: {sorted(desconocidos)}. "
            f"No son coeficientes de {cls.__name__} ni opciones del entorno."
        )

    pobj = cls(**de_dinamica)
    eq = pobj.equilibrium()
    cfg = {
        "parameters": pobj.parameters(),
        "growth_fn": spec["growth_fn"],
        "initial_pop": np.asarray(eq, dtype=np.float32),
        "eq_pop": np.asarray(eq, dtype=np.float64),
    }
    if model != "model_1_2":
        cfg["fished_idx"] = spec["fished_idx"]
    cfg.update(de_entorno)
    return cfg
