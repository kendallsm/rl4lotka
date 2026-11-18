#from rl4lotka import train
#
#options = {
    #"algo": "PPO",                 # "PPO", "RecurrentPPO" o "TQC"
    #"model": "model_1_3",          # "model_1_2", "model_1_3", "model_1_2_1" o "model_2_3"
    #"total_timesteps": 100_000,
    #"save_path": "saved_agents",
    #"config": {},
    #"algo_config": {"policy": "MlpLstmPolicy"},                 
#}

#save_id, options = train(options)

#with open("log.txt", "a", encoding="utf-8") as file:
    #file.write(f"{options}\n")
    #file.write(f"save_id: {save_id}\n\n")

import argparse
import ast

from rl4lotka import train


def valor(texto):
    try:
        return ast.literal_eval(texto)
    except (ValueError, SyntaxError):
        return texto


parser = argparse.ArgumentParser(description="Entrena un agente de rl4lotka")
parser.add_argument("--algo", default="PPO", choices=["PPO", "RecurrentPPO", "TQC"])
parser.add_argument("--model", default="model_1_3",
                    choices=["model_1_2", "model_1_3", "model_1_2_1", "model_2_3"])
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--save-path", default="saved_agents")
parser.add_argument("--id", default=None)
parser.add_argument("--config", nargs="*", default=[], metavar="CLAVE=VALOR")
parser.add_argument("--progress", action="store_true")
parser.add_argument("--checkpoint", type=int, default=0)
args = parser.parse_args()

options = {
    "algo": args.algo,
    "model": args.model,
    "total_timesteps": args.steps,
    "save_path": args.save_path,
    "config": {k: valor(v) for k, v in (c.split("=", 1) for c in args.config)},
}
if args.id:
    options["id"] = args.id
if args.algo == "RecurrentPPO":
    options["algo_config"] = {"policy": "MlpLstmPolicy"}
if args.progress:
    options["progress_bar"] = True
if args.checkpoint:
    options["checkpoint_freq"] = args.checkpoint
save_id, options = train(options)

with open("log.txt", "a", encoding="utf-8") as file:
    file.write(f"{options}\n")
    file.write(f"save_id: {save_id}\n\n")
