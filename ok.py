import sys
import os
import signal
import math
import time
import csv
import argparse
from collections import deque

try:
    import matplotlib
    matplotlib.use("Agg")           # pas besoin d'écran
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib absent — pas de graphe généré")

try:
    from Rosmaster_Lib import Rosmaster
except ImportError:
    print("[ERREUR] Rosmaster_Lib introuvable")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

LOOP_MS    = 20.0
DT         = LOOP_MS / 1000.0

BASE_SPEED = 100.0      # 0-100
MAX_DIFF   = 100.0      # écart max entre côtés
ACCEL_TIME = 3.0        # secondes
BRAKE_TIME = 1.5        # secondes

CSV_FILE   = "straight_line_v8_log.csv"
IMG_FILE   = "straight_line_v8_plot.png"

# ============================================================================
# PID
# ============================================================================

class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integ    = 0.0
        self._prev_e   = 0.0
        self._prev_set = False

    def reset(self):
        self._integ    = 0.0
        self._prev_e   = 0.0
        self._prev_set = False

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(v, hi))

    def compute(self, e, dt):
        self._integ += e * dt
        self._integ  = self._clamp(self._integ, -15.0, 15.0)

        deriv = ((e - self._prev_e) / dt) if self._prev_set else 0.0
        self._prev_e   = e
        self._prev_set = True

        return self.kp * e + self.ki * self._integ + self.kd * deriv

# ============================================================================
# IMU FILTER
# ============================================================================

class YawFilter:
    def __init__(self, alpha=0.30):
        self._alpha = alpha
        self._y     = 0.0
        self._first = True

    @staticmethod
    def _norm(a):
        while a >  180.0: a -= 360.0
        while a < -180.0: a += 360.0
        return a

    def update(self, raw):
        if self._first:
            self._first = False
            self._y = raw
            return self._y
        self._y += self._alpha * self._norm(raw - self._y)
        return self._y

# ============================================================================
# UTILS
# ============================================================================

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def norm_angle(a):
    while a >  180.0: a -= 360.0
    while a < -180.0: a += 360.0
    return a

def read_yaw(bot):
    r, p, y = bot.get_imu_attitude_data(True)
    return y

def wait_imu(bot, timeout=3.0):
    print("[INIT] attente IMU...")
    t0   = time.monotonic()
    prev = 9999.0
    while time.monotonic() - t0 < timeout:
        y = read_yaw(bot)
        if abs(y - prev) > 0.001:
            print("[OK] IMU ACTIVE")
            return True
        prev = y
        time.sleep(0.05)
    return False

def capture_ref(bot, n=50):
    vals = []
    for _ in range(n):
        vals.append(read_yaw(bot))
        time.sleep(0.02)
    return sum(vals) / len(vals)

# def apply_motors(bot, base, diff):
#     """
#     diff > 0 → gauche accélère, droite freine → tourne à droite
#     diff < 0 → droite accélère, gauche freine → tourne à gauche
#     """
#     # left = base
#     # right = base
#     # if diff >= 0:
#     left  = int(round(clamp(base * (1 + diff), -100, 100.0)))
#     # else:
#     right = int(round(clamp(base * (1 - diff), -100, 100.0)))
#     bot.set_motor(left, left, right, right)   # FL, RL, FR, RR
def apply_motors(bot, base, diff):
    """
    Formule additive standard pour entraînement différentiel.
    Garantit une correction maximale même à très basse vitesse (freinage).
    """
    # 1. On applique directement l'ajustement aux deux trains de roues
    # Si le robot roule à base=3.0 mais a besoin d'un diff=-20.0 pour se redresser :
    # La roue gauche passera en marche arrière (-17) pour forcer le pivot.
    left_raw  = base + diff
    right_raw = base - diff
    
    # 2. On sature proprement entre -100 et 100 pour la carte Yahboom
    left_cmd  = int(round(clamp(left_raw, -100.0, 100.0)))
    right_cmd = int(round(clamp(right_raw, -100.0, 100.0)))
    
    # 3. Envoi des commandes aux 4 moteurs (Front-Left, Rear-Left, Front-Right, Rear-Right)
    bot.set_motor(left_cmd, left_cmd, right_cmd, right_cmd)

def stop_robot(bot):
    bot.set_motor(0, 0, 0, 0)
    time.sleep(0.05)

# ============================================================================
# GRAPHE — génération en fin de run ou sur Ctrl+C
# ============================================================================

def generate_plot(rows, out_path=IMG_FILE, kp=None, ki=None, kd=None):
    """
    rows : liste de dict {time, err_deg, diff, left_cmd, right_cmd}
    Génère une image PNG avec 3 sous-graphes.
    """
    if not HAS_MPL or not rows:
        return

    # Ajouter kp/ki/kd au nom du fichier PNG si fournis
    if kp is not None and ki is not None and kd is not None:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}_kp{kp}_ki{ki}_kd{kd}{ext}"

    t       = [r["time"]      for r in rows]
    err     = [r["err_deg"]   for r in rows]
    diff    = [r["diff"]      for r in rows]
    left    = [r["left_cmd"]  for r in rows]
    right   = [r["right_cmd"] for r in rows]

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    # --- Erreur yaw ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, err, color="tab:blue", lw=1.5, label="err_deg")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("Erreur yaw (°)")
    ax1.set_title("Straight Line V8 Differential — Erreur yaw")
    ax1.set_ylim(-3, 3)
    ax1.grid(alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8)

    # --- Différentiel moteur ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, diff, color="tab:orange", lw=1.5, label="diff")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_ylabel("Diff moteur (unités)")
    ax2.set_title("Différentiel de correction")
    ax2.set_ylim(-1.10, 1.10)
    ax2.grid(alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8)

    # --- Commandes moteur gauche/droite ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, left,  color="tab:green",  lw=1.5, label="gauche")
    ax3.plot(t, right, color="tab:red",    lw=1.5, label="droite", ls="--")
    ax3.set_ylabel("Commande moteur (0-100)")
    ax3.set_xlabel("Temps (s)")
    ax3.set_title("Commandes moteurs gauche / droite")
    ax3.set_ylim(-5, 110)
    ax3.grid(alpha=0.35)
    ax3.legend(loc="upper right", fontsize=8)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] graphe sauvegardé → {out_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Straight line V8 differential")
    parser.add_argument("--port",  default="/dev/ttyUSB0")
    parser.add_argument("--speed", type=float, default=60.0)      
    parser.add_argument("--kp",    type=float, default=-3.0)      # Valeur par défaut optimisée en degrés
    parser.add_argument("--ki",    type=float, default=-0.8)      # Intégrale augmentée pour écraser l'erreur à 0
    parser.add_argument("--kd",    type=float, default=-0.25)     # Amortissement adapté
    parser.add_argument("--time",  type=float, default=5.0)
    parser.add_argument("--csv",   default=CSV_FILE)
    parser.add_argument("--img",   default=IMG_FILE)
    args = parser.parse_args()

    print("=== STRAIGHT LINE V8 DIFFERENTIAL ===")
    print(f"BASE_SPEED={args.speed}  MAX_DIFF={MAX_DIFF}")
    print(f"KP={args.kp}  KI={args.ki}  KD={args.kd}\n")

    bot = Rosmaster(1, args.port, 0.002, False)

    rows = []   # données pour le graphe et le CSV

    def handle_signal(sig, frame):
        print("\n[STOP]")
        stop_robot(bot)
        _finalize(rows, args)
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    bot.set_auto_report_state(True, False)
    time.sleep(0.1)
    bot.create_receive_threading()
    time.sleep(0.3)

    if not wait_imu(bot):
        print("[ERREUR] IMU FAILED")
        sys.exit(1)

    yaw_ref = capture_ref(bot)
    print(f"[REF] yaw={yaw_ref:.2f} deg")

    pid = PID(kp=args.kp, ki=args.ki, kd=args.kd)
    yaw_filter = YawFilter(alpha=0.30)

    print("\n[GO] départ dans 2 secondes")
    time.sleep(2.0)

    sum_abs = 0.0
    sum_sq  = 0.0
    max_err = 0.0
    N = int(args.time / DT)
    start = time.monotonic()

    for i in range(N):
        t0      = time.monotonic()
        elapsed = t0 - start

        # --- rampe vitesse ---
        if elapsed < ACCEL_TIME:
            speed = args.speed * (elapsed / ACCEL_TIME)
        elif elapsed > args.time - BRAKE_TIME:
            speed = args.speed * clamp((args.time - elapsed) / BRAKE_TIME, 0.0, 1.0)
        else:
            speed = args.speed

        # --- yaw → erreur géométrique robuste ---
        raw_yaw = read_yaw(bot)
        err_deg = norm_angle(yaw_ref - raw_yaw)

        # --- PID → diff ---
        diff = pid.compute(err_deg, DT)
        diff = clamp(diff, -MAX_DIFF, MAX_DIFF)

        # --- Formule additive unifiée ---
        left_cmd  = speed + diff
        right_cmd = speed - diff

        # --- Compensation de la Zone Morte (Deadzone) ---
        # Empêche les moteurs de siffler sans tourner à cause des frictions mécaniques
        DEADZONE = 10.0
        
        if left_cmd > 0.1 and left_cmd < DEADZONE:       left_cmd = DEADZONE
        elif left_cmd < -0.1 and left_cmd > -DEADZONE:   left_cmd = -DEADZONE
            
        if right_cmd > 0.1 and right_cmd < DEADZONE:     right_cmd = DEADZONE
        elif right_cmd < -0.1 and right_cmd > -DEADZONE: right_cmd = -DEADZONE

        # Saturation finale de sécurité entre -100 et 100
        left_cmd  = clamp(left_cmd, -100.0, 100.0)
        right_cmd = clamp(right_cmd, -100.0, 100.0)

        # --- Envoi direct aux moteurs ---
        left_int  = int(round(left_cmd))
        right_int = int(round(right_cmd))
        bot.set_motor(left_int, left_int, right_int, right_int)

        # --- stats ---
        ae       = abs(err_deg)
        sum_abs += ae
        sum_sq  += ae * ae
        max_err  = max(max_err, ae)

        # --- enregistrement ---
        rows.append({
            "time":      elapsed,
            "err_deg":   err_deg,
            "diff":      diff,
            "left_cmd":  left_cmd,
            "right_cmd": right_cmd,
        })

        # --- log console (toutes les 5 iter = 100ms) ---
        if i % 5 == 0:
            bat = bot.get_battery_voltage()
            print(
                f"t={elapsed:5.2f}"
                f"  yaw={raw_yaw:7.2f}"
                f"  err={err_deg:6.2f}"
                f"  diff={diff:7.2f}"
                f"  L={left_cmd:5.1f}"
                f"  R={right_cmd:5.1f}"
                f"  bat={bat:.2f}"
            )

        # --- timing ---
        used_ms = (time.monotonic() - t0) * 1000.0
        if used_ms < LOOP_MS:
            time.sleep((LOOP_MS - used_ms) / 1000.0)

    stop_robot(bot)
    _finalize(rows, args, N, sum_abs, sum_sq, max_err, yaw_ref, yaw_filter)

def _finalize(rows, args, N=None, sum_abs=0, sum_sq=0, max_err=0,
              yaw_ref=None, yaw_filter=None):
    """Sauvegarde CSV + graphe + stats. Appelé en fin normale ou sur Ctrl+C."""

    # --- CSV ---
    if rows:
        # Ajouter les valeurs de kp/ki/kd au nom du fichier CSV
        base, ext = os.path.splitext(args.csv)
        csv_name = f"{base}_kp{args.kp}_ki{args.ki}_kd{args.kd}{ext}"
        with open(csv_name, "w", newline="") as f:
            # écrire métadonnées en première ligne (commentaire)
            f.write(f"# kp={args.kp}, ki={args.ki}, kd={args.kd}\n")
            w = csv.DictWriter(f, fieldnames=["time","err_deg","diff","left_cmd","right_cmd"])
            w.writeheader()
            w.writerows(rows)
        print(f"[CSV] sauvegardé → {csv_name}")

    # --- graphe ---
    generate_plot(rows, out_path=args.img, kp=args.kp, ki=args.ki, kd=args.kd)

    # --- stats (seulement si run complet) ---
    if N and N > 0 and yaw_filter:
        from Rosmaster_Lib import Rosmaster as _R   # import local pour éviter crash
        mae  = sum_abs / N
        rmse = math.sqrt(sum_sq / N)
        print("\n====================================")
        print("RESULTATS V8 DIFFERENTIAL")
        print("====================================")
        print(f"yaw ref   : {yaw_ref:.2f} deg")
        print(f"MAE       : {mae:.3f} deg")
        print(f"RMSE      : {rmse:.3f} deg")
        print(f"MAX ERROR : {max_err:.3f} deg")
        print(f"base_speed: {args.speed}/100")
        print("====================================")


if __name__ == "__main__":
    main()