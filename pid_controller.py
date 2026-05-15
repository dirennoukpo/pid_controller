import sys
import os
import signal
import math
import time
import csv
import argparse

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("[ERROR] matplotlib est requis pour générer le fichier .png")
    print("        Installez-le avec : pip install matplotlib")
    sys.exit(1)

try:
    from Rosmaster_Lib import Rosmaster
except ImportError:
    print("[ERROR] Rosmaster_Lib not found")
    sys.exit(1)


# ──────────────────────────────────────────────
# PID Controller
# ──────────────────────────────────────────────
class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None, integral_limits=None):
        self.kp              = kp
        self.ki              = ki
        self.kd              = kd
        self.output_limits   = output_limits
        self.integral_limits = integral_limits
        self.integral        = 0.0
        self.previous_error  = 0.0

    def update(self, set_point, current_value, dt):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        error = set_point - current_value

        self.integral += error * dt
        if self.integral_limits is not None:
            lo, hi = self.integral_limits
            self.integral = max(lo, min(hi, self.integral))

        derivative          = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limits is not None:
            lo, hi = self.output_limits
            output = max(lo, min(hi, output))

        return output

    def reset(self):
        self.integral       = 0.0
        self.previous_error = 0.0


# ──────────────────────────────────────────────
# Yaw Filter
# ──────────────────────────────────────────────
class YawFilter:
    def __init__(self, tau: float = 0.08):
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        self._tau   = tau
        self._y     = 0.0
        self._first = True

    @staticmethod
    def _norm(a: float) -> float:
        """Wrap angle difference to [-180, 180]."""
        return (a + 180.0) % 360.0 - 180.0

    def update(self, raw: float, dt: float) -> float:
        if self._first:
            self._y     = raw
            self._first = False
            return self._y
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        alpha    = 1.0 - math.exp(-dt / self._tau)
        self._y += alpha * self._norm(raw - self._y)
        return self._y

    def reset(self, value: float = 0.0) -> None:
        self._y     = value
        self._first = True

    @property
    def value(self) -> float:
        return self._y


# ──────────────────────────────────────────────
# Auto-calibration par encodeurs
# ──────────────────────────────────────────────
def calibrate_motors(bot: Rosmaster, calib_speed: float = 60.0, duration: float = 1.5) -> tuple:
    """
    Fait tourner les 4 moteurs à la même consigne pendant `duration` secondes,
    mesure les ticks d'encodeur produits par chacun, et calcule des facteurs
    correctifs normalisés sur le moteur le plus rapide.

    Layout Rosmaster X3 :
        M1 = Front-Left   M2 = Rear-Left
        M3 = Front-Right  M4 = Rear-Right

    Retourne (f1, f2, f3, f4) où chaque facteur est dans ]0, 1].
    Le moteur le plus rapide reçoit 1.0 ; les autres sont réduits en proportion.
    """
    print("[CAL]  Démarrage de la calibration encodeurs…")

    # Lecture des encodeurs avant
    e1_start, e2_start, e3_start, e4_start = bot.get_motor_encoder()

    # Faire tourner tous les moteurs à la même vitesse
    bot.set_motor(calib_speed, calib_speed, calib_speed, calib_speed)
    time.sleep(duration)
    bot.set_motor(0, 0, 0, 0)
    time.sleep(0.2)  # laisser les moteurs s'arrêter

    # Lecture des encodeurs après
    e1_end, e2_end, e3_end, e4_end = bot.get_motor_encoder()

    ticks = [
        abs(e1_end - e1_start),
        abs(e2_end - e2_start),
        abs(e3_end - e3_start),
        abs(e4_end - e4_start),
    ]

    names = ["M1(FL)", "M2(RL)", "M3(FR)", "M4(RR)"]
    for n, t in zip(names, ticks):
        print(f"[CAL]  {n} → {t} ticks")

    max_ticks = max(ticks)
    if max_ticks == 0:
        print("[CAL]  AVERTISSEMENT : aucun tick lu — encodeurs non fonctionnels, facteurs = 1.0")
        return 1.0, 1.0, 1.0, 1.0

    factors = tuple(t / max_ticks for t in ticks)
    for n, f in zip(names, factors):
        print(f"[CAL]  {n} → facteur = {f:.4f}")

    # Identifier le(s) moteur(s) problématiques (< 90 % du plus rapide)
    threshold = 0.90
    for n, f in zip(names, factors):
        if f < threshold:
            print(f"[CAL]  ⚠  {n} est significativement plus lent ({f:.1%}) — suspect mécanique")

    return factors


# ──────────────────────────────────────────────
# Motor command
# ──────────────────────────────────────────────
def apply_motors(
    bot: Rosmaster,
    base: float,
    diff: float,
    factors: tuple = (1.0, 1.0, 1.0, 1.0),
) -> tuple:
    """
    Compute and send the four motor commands.

    Layout (Rosmaster X3) :
        M1 = Front-Left   M2 = Rear-Left
        M3 = Front-Right  M4 = Rear-Right

    Differential drive :
        left  = base + diff
        right = base - diff

    Les facteurs de calibration compensent les déséquilibres mécaniques
    moteur par moteur, avant le clamping final.

    Returns (fl, rl, fr, rr) as actually sent.
    """
    f1, f2, f3, f4 = factors

    left  = base + diff
    right = base - diff

    fl = max(-100.0, min(100.0, left  * f1))
    rl = max(-100.0, min(100.0, left  * f2))
    fr = max(-100.0, min(100.0, right * f3))
    rr = max(-100.0, min(100.0, right * f4))

    bot.set_motor(fl, rl, fr, rr)
    return fl, rl, fr, rr


# ──────────────────────────────────────────────
# Graph + CSV
# ──────────────────────────────────────────────
def _finalize(rows: list, args, factors: tuple) -> None:
    if not rows:
        return

    tag = f"kp{args.kp}_ki{args.ki}_kd{args.kd}"

    # ── CSV ────────────────────────────────────
    csv_name = f"run_{tag}.csv"
    with open(csv_name, "w", newline="") as f:
        f.write(f"# kp={args.kp}  ki={args.ki}  kd={args.kd}  base={args.base}\n")
        f.write(f"# calib_factors  f1={factors[0]:.4f}  f2={factors[1]:.4f}  f3={factors[2]:.4f}  f4={factors[3]:.4f}\n")
        w = csv.DictWriter(f, fieldnames=["time", "yaw", "err_deg", "diff", "fl", "rl", "fr", "rr"])
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV]  → {csv_name}")

    # ── Stats ──────────────────────────────────
    errors = [abs(r["err_deg"]) for r in rows]
    n      = len(errors)
    mae    = sum(errors) / n
    rmse   = math.sqrt(sum(e * e for e in errors) / n)
    maxe   = max(errors)
    print(f"[STAT] MAE={mae:.3f}°  RMSE={rmse:.3f}°  MAX={maxe:.3f}°  n={n}")

    # ── Plot ───────────────────────────────────
    png_name = f"run_{tag}.png"
    t    = [r["time"]    for r in rows]
    err  = [r["err_deg"] for r in rows]
    diff = [r["diff"]    for r in rows]
    fl   = [r["fl"]      for r in rows]
    fr   = [r["fr"]      for r in rows]

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        f"kp={args.kp}  ki={args.ki}  kd={args.kd}  base={args.base}"
        f"\ncalib: f1={factors[0]:.3f}  f2={factors[1]:.3f}  f3={factors[2]:.3f}  f4={factors[3]:.3f}",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, err, color="tab:blue", lw=1.5, label="erreur yaw (°)")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("Erreur (°)")
    ax1.set_title("Erreur yaw")
    ax1.grid(alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, diff, color="tab:orange", lw=1.5, label="diff (correction PID)")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_ylabel("Diff")
    ax2.set_title("Correction différentielle")
    ax2.grid(alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, fl, color="tab:green", lw=1.5, label="FL / RL (gauche)")
    ax3.plot(t, fr, color="tab:red",   lw=1.5, label="FR / RR (droite)", ls="--")
    ax3.set_ylabel("Commande (0–100)")
    ax3.set_xlabel("Temps (s)")
    ax3.set_title("Commandes moteurs")
    ax3.grid(alpha=0.35)
    ax3.legend(loc="upper right", fontsize=8)

    plt.savefig(png_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] → {png_name}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
_running = True

def _handle_signal(sig, frame):
    global _running
    _running = False


def main():
    global _running

    parser = argparse.ArgumentParser(description="Straight-line PID controller — Yahboom Rosmaster")
    parser.add_argument("--port",        type=str,   default="/dev/ttyUSB0", help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--base",        type=float, default=85.0,            help="Forward speed 0-100 (default: 85)")
    parser.add_argument("--duration",    type=float, default=5.0,             help="Run duration in seconds (default: 5)")
    parser.add_argument("--kp",          type=float, default=5.8)
    parser.add_argument("--ki",          type=float, default=4.35)
    parser.add_argument("--kd",          type=float, default=0.1005)
    parser.add_argument("--no-calib",    action="store_true",                 help="Désactiver la calibration encodeurs")
    parser.add_argument("--calib-speed", type=float, default=100.0,            help="Vitesse de calibration 0-100 (default: 60)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    DT   = 0.02  # 50 Hz control loop
    rows = []

    # ── Init robot ──────────────────────────────────────────────────────
    print(f"[INFO] Port série : {args.port}")
    bot = Rosmaster(car_type=1, com=args.port)
    bot.create_receive_threading()
    time.sleep(0.5)

    # ── Calibration encodeurs ───────────────────────────────────────────
    if args.no_calib:
        factors = (1.0, 1.0, 1.0, 1.0)
        print("[CAL]  Calibration désactivée — facteurs = 1.0 pour tous les moteurs")
    else:
        factors = calibrate_motors(bot, calib_speed=args.calib_speed, duration=3.0)
        print("[CAL]  Attente 5 secondes avant le départ…")
        time.sleep(5.0)

    pid  = PIDController(
        kp=-args.kp, ki=-args.ki, kd=-args.kd,
        output_limits=(-args.base, args.base),
        integral_limits=(-15.0, 15.0),
    )
    filt = YawFilter(tau=0.08)

    # ── Capture de la référence de cap ─────────────────────────────────
    _, _, raw_ref = bot.get_imu_attitude_data(ToAngle=True)
    yaw_ref = filt.update(raw_ref, dt=DT)
    print(f"[INFO] Référence yaw : {yaw_ref:.2f}°")
    print(f"[INFO] Durée : {args.duration:.1f}s  |  Base : {args.base}  |  PID kp={-args.kp} ki={-args.ki} kd={-args.kd}")

    t_start = time.monotonic()
    t_end   = t_start + args.duration
    t_prev  = t_start

    while _running:
        now = time.monotonic()
        if now >= t_end:
            break

        dt_real = max(now - t_prev, 1e-4)
        t_prev  = now
        elapsed = now - t_start

        _, _, raw_yaw = bot.get_imu_attitude_data(ToAngle=True)
        filtered_yaw  = filt.update(raw_yaw, dt=dt_real)

        diff = pid.update(yaw_ref, filtered_yaw, dt=dt_real)

        fl, rl, fr, rr = apply_motors(bot, args.base, diff, factors)

        rows.append({
            "time":    elapsed,
            "yaw":     filtered_yaw,
            "err_deg": yaw_ref - filtered_yaw,
            "diff":    diff,
            "fl": fl, "rl": rl, "fr": fr, "rr": rr,
        })

        print(
            f"t={elapsed:5.2f}s  yaw={filtered_yaw:7.2f}°"
            f"  err={yaw_ref - filtered_yaw:+6.2f}°"
            f"  diff={diff:+6.2f}"
            f"  FL={fl:+5.1f}  RL={rl:+5.1f}  FR={fr:+5.1f}  RR={rr:+5.1f}"
        )

        sleep_for = DT - (time.monotonic() - now)
        if sleep_for > 0:
            time.sleep(sleep_for)

    # ── Arrêt propre ───────────────────────────────────────────────────
    apply_motors(bot, base=0.0, diff=0.0, factors=factors)
    print("[INFO] Arrêt.")
    _finalize(rows, args, factors)


if __name__ == "__main__":
    main()