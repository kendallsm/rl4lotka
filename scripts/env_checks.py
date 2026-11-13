import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import rl4lotka  # noqa: F401  (registra los ambientes)

model = sys.argv[1] if len(sys.argv) > 1 else "model_1_2"
n_runs, n_steps, burn_in = 20, 1000, 500

env = gym.make(model, config={"Tmax": n_steps})
species = env.unwrapped.species_names
eq_pop = env.unwrapped.eq_pop

trajectories, collapses = [], 0
for run in range(n_runs):
    env.reset(seed=run)
    pops = [env.unwrapped.population()]
    for t in range(n_steps):
        _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        pops.append(info["population"])
        if terminated or truncated:
            break
    collapses += terminated
    trajectories.append(np.array(pops))

finished = [tr for tr in trajectories if len(tr) > burn_in]
means = np.array([tr[burn_in:].mean(axis=0) for tr in finished])
print(f"{model}: {n_runs} corridas sin pesca de {n_steps} pasos")
print(f"  colapsan: {collapses}/{n_runs}")
if len(means):
    print(f"  promedio (despues de {burn_in} pasos): {np.round(means.mean(axis=0), 4)}  especies {species}")
    print(f"  dispersion entre corridas:               {np.round(means.std(axis=0), 4)}")
print(f"  equilibrio de referencia (recompensa):   {np.round(eq_pop, 4)}")

fig, ax = plt.subplots(figsize=(9, 4))
for i, name in enumerate(species):
    for k, tr in enumerate(trajectories[:5]):
        ax.plot(tr[:, i], color=f"C{i}", alpha=0.6, lw=0.8, label=name if k == 0 else None)
    ax.axhline(eq_pop[i], color=f"C{i}", ls="--", lw=1)
ax.set_xlabel("paso")
ax.set_ylabel("poblacion")
ax.set_title(f"{model} sin pesca (5 corridas; lineas punteadas = equilibrio de referencia)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{model}_sin_pesca.png", dpi=120)
print(f"  figura: {model}_sin_pesca.png")
