import argparse
import glob
import os

import matplotlib as mpl
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="csv")
parser.add_argument("--out", default="figures")
parser.add_argument("--model", default=None)
args = parser.parse_args()

mpl.rcParams["axes.linewidth"] = 2
sns.set_theme(palette=sns.color_palette("colorblind"), font_scale=1.4, style="whitegrid")
os.makedirs(args.out, exist_ok=True)

archivos = sorted(glob.glob(os.path.join(args.csv, "*_rew.csv")))
if args.model:
    archivos = [a for a in archivos if os.path.basename(a) == f"{args.model}_rew.csv"]

for archivo in archivos:
    modelo = os.path.basename(archivo).replace("_rew.csv", "")
    df = pd.read_csv(archivo)

    # kde necesita varianza; si todos los episodios dan lo mismo se usa histograma
    varia = df.groupby(["scenario", "policy_type", "agent"])["rew"].std().fillna(0).min() > 1e-9
    g = sns.displot(
        df, x="rew", hue="agent", kind="kde" if varia else "hist",
        fill=True, row="policy_type", col="scenario",
        facet_kws=dict(margin_titles=True, sharey=False, sharex="col"),
        **({} if varia else {"element": "step", "stat": "density"}),
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name} policies")
    g.set_axis_labels("Recompensa del episodio", "")
    g.figure.suptitle(modelo, y=1.02)
    salida = os.path.join(args.out, f"{modelo}_rewards.png")
    g.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"{salida}")
