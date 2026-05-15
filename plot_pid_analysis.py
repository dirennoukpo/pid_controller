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
import math
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as mcm
except ImportError:
    print("[ERROR] matplotlib requis : pip install matplotlib")
    sys.exit(1)


# ──────────────────────────────────────────────
# Tolérance pour comparaison flottante
# ──────────────────────────────────────────────
FLOAT_TOL = 1e-9


def feq(a: float, b: float) -> bool:
    """Égalité flottante avec tolérance relative+absolue."""
    return math.isclose(a, b, rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)


# ──────────────────────────────────────────────
# Lecture d'un CSV de run
# ──────────────────────────────────────────────
def load_csv(path: str, pid_only: bool = False):
    """
    Retourne (kp, ki, kd, times, errors) depuis un fichier run CSV.
    Si pid_only=True, filtre sur la colonne 'phase'=='pid'.
    Retourne None si le fichier est invalide.
    """
    # Extraire kp, ki, kd depuis le nom de fichier
    # Accepte des valeurs comme 0.05, 1e-3, 1.5, 10, etc.
    m = re.search(
        r"kp([^_]+)_ki([^_]+)_kd([^_.]+(?:\.[0-9]+)?)",
        os.path.basename(path)
    )
    if not m:
        print(f"[WARN] Nom de fichier non reconnu (kp/ki/kd introuvables) : {os.path.basename(path)}")
        return None

    try:
        kp = float(m.group(1))
        ki = float(m.group(2))
        kd = float(m.group(3))
    except ValueError as e:
        print(f"[WARN] Impossible de parser kp/ki/kd dans {os.path.basename(path)} : {e}")
        return None

    times  = []
    errors = []

    try:
        with open(path, newline="") as f:
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        fieldnames = reader.fieldnames or []
        has_phase = "phase" in fieldnames

        if pid_only and not has_phase:
            print(f"[WARN] --pid-only actif mais colonne 'phase' absente dans {os.path.basename(path)} — toutes les lignes incluses")

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
        print(f"[WARN] Aucun point de données valide dans {os.path.basename(path)}")
        return None

    return kp, ki, kd, times, errors


# ──────────────────────────────────────────────
# Valeurs uniques d'une liste de floats (avec tolérance)
# ──────────────────────────────────────────────
def unique_values(values: list) -> list:
    """Retourne les valeurs uniques (avec tolérance flottante), triées."""
    uniq = []
    for v in values:
        if not any(feq(v, u) for u in uniq):
            uniq.append(v)
    return sorted(uniq)


# ──────────────────────────────────────────────
# Construction des groupes de comparaison
# ──────────────────────────────────────────────
def best_refs_for(runs: list, vary: str) -> tuple:
    """
    Pour le paramètre `vary` ('kp', 'ki' ou 'kd'), trouve la combinaison
    des deux paramètres fixes qui maximise le nombre de courbes traçables
    (i.e. le nombre de valeurs distinctes du paramètre variable).

    Retourne (ref_a, ref_b, group) où :
      - ref_a, ref_b  : valeurs fixes choisies
      - group         : [(vary_val, times, errors), ...]
    """
    IDX = {"kp": 0, "ki": 1, "kd": 2}
    fixed = [p for p in ("kp", "ki", "kd") if p != vary]
    fa, fb = fixed[0], fixed[1]
    ia, ib, iv = IDX[fa], IDX[fb], IDX[vary]

    vals_a = unique_values([r[ia] for r in runs])
    vals_b = unique_values([r[ib] for r in runs])

    best_count = -1
    best_a = best_b = None
    best_group = []

    for va in vals_a:
        for vb in vals_b:
            candidates = [
                (r[iv], r[3], r[4])
                for r in runs
                if feq(r[ia], va) and feq(r[ib], vb)
            ]
            distinct = len(unique_values([c[0] for c in candidates]))
            if distinct > best_count:
                best_count = distinct
                best_a = va
                best_b = vb
                best_group = candidates

    best_group.sort(key=lambda x: x[0])
    return best_a, best_b, best_group


def build_groups(runs: list) -> tuple:
    """
    runs : liste de (kp, ki, kd, times, errors)

    Pour chaque axe de variation, sélectionne indépendamment les valeurs
    de référence des deux autres paramètres qui maximisent le nombre de
    courbes affichées.

    Retourne (groups, refs) où :
      groups = {"kp": [(val, times, errors), ...], "ki": [...], "kd": [...]}
      refs   = {"kp": {"ki": v, "kd": v}, "ki": {"kp": v, "kd": v}, "kd": {"kp": v, "ki": v}}
    """
    groups = {}
    refs   = {}

    for vary in ("kp", "ki", "kd"):
        fixed = [p for p in ("kp", "ki", "kd") if p != vary]
        fa, fb = fixed[0], fixed[1]
        ref_a, ref_b, group = best_refs_for(runs, vary)
        groups[vary] = group
        refs[vary]   = {fa: ref_a, fb: ref_b}
        print(f"[INFO] Groupe {vary.upper():2s} : {len(group)} courbe(s)  "
              f"[{fa}={ref_a}  {fb}={ref_b}]")

    # Runs absents de tous les groupes
    IDX = {"kp": 0, "ki": 1, "kd": 2}
    classified = set()
    for vary in ("kp", "ki", "kd"):
        fa, fb = [p for p in ("kp", "ki", "kd") if p != vary]
        for val, t, e in groups[vary]:
            for r in runs:
                if (feq(r[IDX[vary]], val)
                        and feq(r[IDX[fa]], refs[vary][fa])
                        and feq(r[IDX[fb]], refs[vary][fb])):
                    classified.add((r[0], r[1], r[2]))

    unclassified = [
        (r[0], r[1], r[2]) for r in runs
        if (r[0], r[1], r[2]) not in classified
    ]
    if unclassified:
        print(f"\n[WARN] {len(unclassified)} run(s) absent(s) de tous les groupes :")
        for kp, ki, kd in unclassified:
            print(f"       kp={kp}  ki={ki}  kd={kd}")

    return groups, refs


# ──────────────────────────────────────────────
# Récupération d'une colormap (compatible toutes versions matplotlib)
# ──────────────────────────────────────────────
def get_cmap_compat(name: str):
    """Compatibilité matplotlib <3.7 et >=3.7."""
    try:
        return mcm.get_cmap(name)          # <3.7
    except AttributeError:
        return mcm.colormaps[name]          # >=3.7


# ──────────────────────────────────────────────
# Tracé d'un sous-graphe
# ──────────────────────────────────────────────
def plot_group(ax, group: list, coeff_name: str, ref_pair: str, cmap_name: str = "plasma"):
    """
    group : [(coeff_val, times, errors), ...]
    """
    if not group:
        ax.text(
            0.5, 0.5,
            f"Aucune donnée\npour faire varier {coeff_name}\n\n"
            f"(aucun run avec les valeurs de référence\n{ref_pair})",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray"
        )
        ax.set_title(f"Impact de {coeff_name}  ({ref_pair})", fontsize=10, fontweight="bold")
        return

    cmap   = get_cmap_compat(cmap_name)
    n      = len(group)
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
    parser.add_argument("--dir",      type=str,  default=".",               help="Répertoire contenant les CSV (défaut: .)")
    parser.add_argument("--out",      type=str,  default="pid_analysis.png", help="Fichier PNG de sortie")
    parser.add_argument("--pid-only", action="store_true",                  help="N'afficher que la phase PID")
    args = parser.parse_args()

    # ── Recherche des CSV ───────────────────────────────────────────────
    pattern = os.path.join(args.dir, "run_kp*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"[ERROR] Aucun fichier run_kp*.csv trouvé dans : {os.path.abspath(args.dir)}")
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
        runs.append((kp, ki, kd, times, errors))
        print(f"[OK]   kp={kp}  ki={ki}  kd={kd}  —  {len(times)} points")

    if not runs:
        print("[ERROR] Aucune donnée valide à tracer.")
        sys.exit(1)

    print(f"\n[INFO] {len(runs)}/{len(csv_files)} fichier(s) chargé(s) avec succès.")

    # ── Groupes de comparaison ───────────────────────────────────────────
    groups, refs = build_groups(runs)

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
        ref_pair=f"Ki={refs['kp']['ki']}  Kd={refs['kp']['kd']}",
        cmap_name="plasma"
    )
    plot_group(
        ax_ki, groups["ki"], "Ki",
        ref_pair=f"Kp={refs['ki']['kp']}  Kd={refs['ki']['kd']}",
        cmap_name="viridis"
    )
    plot_group(
        ax_kd, groups["kd"], "Kd",
        ref_pair=f"Kp={refs['kd']['kp']}  Ki={refs['kd']['ki']}",
        cmap_name="inferno"
    )

    out_path = os.path.join(args.dir, args.out)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT] → {out_path}")


if __name__ == "__main__":
    main()