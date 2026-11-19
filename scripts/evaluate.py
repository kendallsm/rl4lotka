import argparse
import ast
import os

import gymnasium as gym
import pandas as pd

from rl4lotka import envs  # noqa: F401  (registra los entornos)
from rl4lotka.evaluate import FixedEffort, evaluate_agent
from rl4lotka.train.sb3 import algorithm


def leer_log(path):
    """Devuelve la lista de options guardadas en log.txt (con save_id)."""
    corridas = {}
    with open(path, encoding="utf-8") as f:
        bloques = f.read().strip().split("\n\n")
    for bloque in bloques:
        lineas = [l for l in bloque.splitlines() if l.strip()]
        if len(lineas) < 2 or not lineas[-1].startswith("save_id:"):
            continue
        options = ast.literal_eval(lineas[0])
        save_id = lineas[-1].split(":", 1)[1].strip()
        options["save_id"] = save_id
        corridas[save_id] = options          # si se repite, queda la ultima
    return list(corridas.values())


def crear_env(options):
    return gym.make(options["model"], config=options.get("config", {})).unwrapped


def evaluar(agent, env, nombre, tipo, escenario, n, ray_remote):
    eps = evaluate_agent(agent=agent, env=env, ray_remote=ray_remote).episodes(n)
    df = pd.DataFrame(eps)
    df.insert(0, "ep", range(len(df)))
    df["agent"] = nombre
    df["policy_type"] = tipo
    df["scenario"] = escenario
    return df


parser = argparse.ArgumentParser(description="Evalua los agentes de log.txt")
parser.add_argument("--log", default="log.txt")
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--model", default=None, help="solo este modelo")
parser.add_argument("--out", default="csv")
parser.add_argument("--efforts", nargs="*", type=float, default=[0.0, 0.1, 0.3],
                    help="esfuerzos constantes de referencia (0 = sin pesca)")
parser.add_argument("--ray", action="store_true", help="paralelo con ray")
args = parser.parse_args()

corridas = leer_log(args.log)
if args.model:
    corridas = [c for c in corridas if c["model"] == args.model]
if not corridas:
    raise SystemExit(f"no encontre corridas en {args.log}")

os.makedirs(args.out, exist_ok=True)
por_modelo = {}
for opt in corridas:
    if not os.path.exists(opt["save_id"]):
        print(f"[salto] no existe {opt['save_id']}")
        continue
    escenario = opt.get("id") or "default"
    env = crear_env(opt)
    agent = algorithm(opt["algo"]).load(opt["save_id"], device="cpu")
    print(f"evaluando {opt['save_id']}")
    df = evaluar(agent, env, opt["algo"], "RL", escenario, args.episodes, args.ray)
    por_modelo.setdefault(opt["model"], []).append(df)

    # referencias fijas, una vez por (modelo, escenario)
    clave = (opt["model"], escenario)
    if clave not in por_modelo.setdefault("_hechos", set()):
        por_modelo["_hechos"].add(clave)
        for e in args.efforts:
            nombre = "Sin pesca" if e == 0 else f"Esfuerzo {e:g}"
            ref = evaluar(FixedEffort(env, e), env, nombre, "Fixed", escenario,
                          args.episodes, args.ray)
            por_modelo[opt["model"]].append(ref)

por_modelo.pop("_hechos", None)
for modelo, dfs in por_modelo.items():
    tabla = pd.concat(dfs, ignore_index=True)
    salida = os.path.join(args.out, f"{modelo}_rew.csv")
    tabla.to_csv(salida, index=False)
    print(f"\n{modelo} -> {salida}")
    resumen = (tabla.groupby(["scenario", "agent"])
                    .agg(rew_media=("rew", "mean"), rew_std=("rew", "std"),
                         largo_medio=("length", "mean"),
                         dist_eq_final=("eq_distance", "mean"))
                    .round(2))
    print(resumen)
