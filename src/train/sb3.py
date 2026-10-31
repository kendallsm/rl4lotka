import importlib
import os

import torch
import yaml
from sb3_contrib import ARS, TQC, RecurrentPPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import envs  # noqa: F401
ALGOS = {
    "ppo": PPO,
    #  vienen de sb3_contrib
    "tqc": TQC,
    "recurrentppo": RecurrentPPO,  
    "rppo": RecurrentPPO,
}
#Para leer los yml
def load_options(config_file, **kwargs): 
    with open(config_file, "r", encoding="utf-8") as f:
        options = yaml.safe_load(f)
    return {**options, **kwargs}


def model_name(options):
    #PPO-model12-prueba, asi seria el nombre del archivo
    name = f"{options['algo']}-{options['model']}"
    if options.get("id"):
        name += f"-{options['id']}"
    return name



#

def _parse_policy_kwargs(algo_config, additional_imports=()):
    #texto a diccionario
        #policy_kwargs: "dict(net_arch=[64, 32, 16])"
    
    pk = algo_config.get("policy_kwargs")
    if not isinstance(pk, str):
        return algo_config  # ya es un dict, o no hay policy_kwargs

    #esto es para del torch.nn
    namespace = {"dict": dict, "torch": torch}
    for name in additional_imports:
        print(f"importing {name}")
        importlib.import_module(name)               # importa torch.nn
        top = name.split(".")[0]                    # pasa "torch.nn" a "torch"
        namespace[top] = importlib.import_module(top)

    algo_config["policy_kwargs"] = eval(pk, namespace)
    return algo_config


#hacemos el ambiente
def make_env(options):
    return make_vec_env(
        options["model"],  # nombre del ambiente ,  "model12"
        n_envs=options.get("n_envs", 1),
        seed=options.get("seed"),
        env_kwargs={"config": options.get("config", {})},
    )

#entrenamos

def sb3_train(config_file, **kwargs):
    #entrena el agente con lo que le di del yml
    options = load_options(config_file, **kwargs)
    return train(options)


def train(options):
    #entrena el agente cuando le meto un diccionario de opciones
    env = make_env(options)

    # Copia de algo_config para no modificar el dict original.
    algo_config = dict(options.get("algo_config", {}))
    algo_config.setdefault("policy", "MlpPolicy")  # red neuronal estándar
    algo_config = _parse_policy_kwargs(
        algo_config, options.get("additional_imports", [])
    )

    ALGO = algorithm(options["algo"])
    model = ALGO(env=env, **algo_config)

    name = model_name(options)
    save_path = options["save_path"]
    # se crea la carpeta antes de iniciar no despues
    os.makedirs(save_path, exist_ok=True)

    # para guardar las copias mientras entreba
    callback = None
    if options.get("checkpoint_freq"):
        # para contar pasos
        callback = CheckpointCallback(
            save_freq=max(options["checkpoint_freq"] // env.num_envs, 1),
            save_path=save_path,
            name_prefix=name,  # archivos: <name>_<pasos>_steps.zip
        )

    model.learn(
        total_timesteps=options["total_timesteps"],
        tb_log_name=name,  # nombre de la corrida en tensorboard
        progress_bar=options.get("progress_bar", False),  # requiere tqdm y rich
        callback=callback,
    )

    save_id = os.path.join(save_path, name + ".zip")
    model.save(save_id)
    print(f"Saved {options['algo']} model at {save_id}")

    env.close()
    return save_id, options


#cargar agente
def load_agent(path, algo, env=None):
  #por si hay agentes guardados
    return algorithm(algo).load(path, env=env)