from rl4lotka import train

options = {
    "algo": "PPO",                 # "PPO", "RecurrentPPO" o "TQC"
    "model": "model_1_3",          # "model_1_2", "model_1_3", "model_1_2_1" o "model_2_3"
    "total_timesteps": 100_000,
    "config": {},                  # cambios a params.py o al entorno, p. ej. {"s_D": 0.1, "Tmax": 500}
}

save_id, options = train(options)

with open("log.txt", "a", encoding="utf-8") as file:
    file.write(f"{options}\n")
    file.write(f"save_id: {save_id}\n\n")
