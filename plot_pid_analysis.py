"""
plot_pid_analysis.py
────────────────────
Lit tous les fichiers run_kpX_kiY_kdZ.csv présents dans le répertoire
courant et génère un fichier pid_analysis.png contenant 3 graphes :

  Graphe 1 — Impact de Kp  : toutes les runs où Ki et Kd sont identiques
                              (valeur de référence = la plus fréquente)
  Graphe 2 — Impact de Ki  : idem, Ki varie, Kp et Kd fixes
  Graphe 3 — Impact de Kd  : idem, Kd varie, Kp et Ki fixes

Chaque courbe représente l'erreur yaw (°) en fonction du temps pour un
run donné.  La légende indique la valeur du coefficient qui varie.

Usage :
    python3 plot_pid_analysis.py [--dir ./] [--out pid_analysis.png] [--pid-only]

Options :
    --dir       Répertoire contenant les CSV  (défaut : répertoire courant)
    --out       Nom du fichier PNG de sortie  (défaut : pid_analysis.png)
    --pid-only  N'afficher que la phase PID   (ignore la phase calibration)
"""

import sys
import os
import csv
import argparse
import glob
import re
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.cm import get_cmap
except ImportError:
    print("[ERROR] matplotlib requis : pip install matplotlib")
    sys.exit(1)


# ──────────────────────────────────────────────
# Lecture d'un CSV de run
# ──────────────────────────────────────────────
def load_csv(path: str, pid_only: bool = False) -> tuple:
    """
    Retourne (kp, ki, kd, times, errors) depuis un fichier run CSV.
    Si pid_only=True, filtre sur la colonne 'phase'=='pid'.
    Retourne None si le fichier est invalide.
    """
    kp = ki = kd = None

    # Extraire kp, ki, kd depuis le nom de fichier
    m = re.search(r"kp([0-9eE.+-]+)_ki([0-9eE.+-]+)_kd([0-9eE.+-]+?)(?:\.csv)?$", os.path.basename(path))
    if m:
        kp = float(m.group(1))
        ki = float(m.group(2))
        kd = float(m.group(3))

    times  = []
    errors = []

    try:
        with open(path, newline="") as f:
            # Sauter les lignes de commentaires (#)
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        has_phase = "phase" in (reader.fieldnames or [])

        for row in reader:
            if pid_only and has_phase and row.get("phase") != "pid":
                continue
            try:
                times.append(float(row["time"]))
                errors.append(float(row["err_deg"]))
            except (KeyError, ValueError):
                continue
    except Exception as e:
        print(f"[WARN] Impossible de lire {path} : {e}")
        return None

    if not times:
        return None

    return kp, ki, kd, times, errors


# ──────────────────────────────────────────────
# Valeur modale (la plus fréquente dans une liste)
# ──────────────────────────────────────────────
def modal_value(values: list):
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    return max(counts, key=counts.__getitem__)


# ──────────────────────────────────────────────
# Construction des groupes de comparaison
# ──────────────────────────────────────────────
def build_groups(runs: list) -> dict:
    """
    runs : liste de (kp, ki, kd, times, errors)

    Retourne un dict :
    {
      "kp": [(kp_val, ki_ref, kd_ref, times, errors), ...],
      "ki": [(kp_ref, ki_val, kd_ref, times, errors), ...],
      "kd": [(kp_ref, ki_ref, kd_val, times, errors), ...],
    }

    Pour chaque coefficient, la valeur de référence des deux autres
    est choisie comme la valeur la plus fréquente dans l'ensemble des runs.
    """
    kp_vals = [r[0] for r in runs]
    ki_vals = [r[1] for r in runs]
    kd_vals = [r[2] for r in runs]

    ref_kp = modal_value(kp_vals)
    ref_ki = modal_value(ki_vals)
    ref_kd = modal_value(kd_vals)

    print(f"[INFO] Valeurs de référence : kp={ref_kp}  ki={ref_ki}  kd={ref_kd}")

    groups = {"kp": [], "ki": [], "kd": []}

    for kp, ki, kd, t, e in runs:
        # Groupe kp : ki == ref_ki ET kd == ref_kd
        if ki == ref_ki and kd == ref_kd:
            groups["kp"].append((kp, t, e))
        # Groupe ki : kp == ref_kp ET kd == ref_kd
        if kp == ref_kp and kd == ref_kd:
            groups["ki"].append((ki, t, e))
        # Groupe kd : kp == ref_kp ET ki == ref_ki
        if kp == ref_kp and ki == ref_ki:
            groups["kd"].append((kd, t, e))

    # Tri par valeur du coefficient variable
    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    return groups, ref_kp, ref_ki, ref_kd


# ──────────────────────────────────────────────
# Tracé d'un sous-graphe
# ──────────────────────────────────────────────
def plot_group(ax, group: list, coeff_name: str, ref_pair: str, cmap_name: str = "plasma"):
    """
    group : [(coeff_val, times, errors), ...]
    """
    if not group:
        ax.text(0.5, 0.5, f"Aucune donnée\npour faire varier {coeff_name}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title(f"Impact de {coeff_name}  ({ref_pair})")
        return

    cmap   = get_cmap(cmap_name)
    n      = max(len(group), 1)
    colors = [cmap(i / (n - 1) if n > 1 else 0.5) for i in range(n)]

    for idx, (val, times, errors) in enumerate(group):
        label = f"{coeff_name}={val}"
        ax.plot(times, errors, color=colors[idx], lw=1.6, label=label, alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Erreur yaw (°)", fontsize=9)
    ax.set_xlabel("Temps (s)", fontsize=9)
    ax.set_title(f"Impact de {coeff_name}  —  {ref_pair}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyse comparative PID — génère pid_analysis.png")
    parser.add_argument("--dir",      type=str,  default=".",              help="Répertoire contenant les CSV (défaut: .)")
    parser.add_argument("--out",      type=str,  default="pid_analysis.png", help="Fichier PNG de sortie")
    parser.add_argument("--pid-only", action="store_true",                 help="N'afficher que la phase PID")
    args = parser.parse_args()

    # ── Recherche des CSV ───────────────────────────────────────────────
    pattern = os.path.join(args.dir, "run_kp*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"[ERROR] Aucun fichier run_kp*.csv trouvé dans : {args.dir}")
        sys.exit(1)

    print(f"[INFO] {len(csv_files)} fichier(s) CSV trouvé(s) :")
    for f in csv_files:
        print(f"       {os.path.basename(f)}")

    # ── Chargement ──────────────────────────────────────────────────────
    runs = []
    for path in csv_files:
        result = load_csv(path, pid_only=args.pid_only)
        if result is None:
            print(f"[WARN] Ignoré : {os.path.basename(path)}")
            continue
        kp, ki, kd, times, errors = result
        if kp is None:
            print(f"[WARN] kp/ki/kd non détectés dans le nom de fichier : {os.path.basename(path)}")
            continue
        runs.append((kp, ki, kd, times, errors))
        print(f"[OK]   kp={kp}  ki={ki}  kd={kd}  —  {len(times)} points")

    if not runs:
        print("[ERROR] Aucune donnée valide à tracer.")
        sys.exit(1)

    # ── Groupes de comparaison ───────────────────────────────────────────
    groups, ref_kp, ref_ki, ref_kd = build_groups(runs)

    # ── Tracé ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 13))

    phase_label = " (phase PID uniquement)" if args.pid_only else ""
    fig.suptitle(
        f"Analyse comparative des coefficients PID{phase_label}\n"
        f"({len(runs)} run(s) chargé(s))",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.55, top=0.91, bottom=0.06)

    ax_kp = fig.add_subplot(gs[0])
    ax_ki = fig.add_subplot(gs[1])
    ax_kd = fig.add_subplot(gs[2])

    plot_group(
        ax_kp, groups["kp"], "Kp",
        ref_pair=f"Ki={ref_ki}  Kd={ref_kd}",
        cmap_name="plasma"
    )
    plot_group(
        ax_ki, groups["ki"], "Ki",
        ref_pair=f"Kp={ref_kp}  Kd={ref_kd}",
        cmap_name="viridis"
    )
    plot_group(
        ax_kd, groups["kd"], "Kd",
        ref_pair=f"Kp={ref_kp}  Ki={ref_ki}",
        cmap_name="inferno"
    )

    out_path = os.path.join(args.dir, args.out)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] → {out_path}")


if __name__ == "__main__":
    main()