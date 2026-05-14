class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None, integral_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits        # (min, max) or None
        self.integral_limits = integral_limits    # (min, max) or None — windup guard

        self.integral = 0
        self.previous_error = 0

    def update(self, set_point, current_value, dt):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        error = set_point - current_value

        # Integral with optional windup clamp
        self.integral += error * dt
        if self.integral_limits is not None:
            lo, hi = self.integral_limits
            self.integral = max(lo, min(hi, self.integral))

        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Optional output clamp
        if self.output_limits is not None:
            lo, hi = self.output_limits
            output = max(lo, min(hi, output))

        return output

    def reset(self):
        self.integral = 0
        self.previous_error = 0

import math

class YawFilter:
    """
    Exponential low-pass filter for yaw angle (degrees, [-180, 180]).

    Parameters
    ----------
    alpha : float, optional
        Fixed smoothing factor in (0, 1]. Used when dt is not provided.
        alpha=1.0 → no filtering. alpha→0 → very heavy filtering.
    tau : float, optional
        Time constant (seconds). When provided alongside dt in update(),
        alpha is recomputed each step as 1 - exp(-dt / tau),
        making the filter independent of sampling rate.
    """

    def __init__(self, alpha: float = 0.30, tau: float | None = None):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if tau is not None and tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")

        self._alpha = alpha
        self._tau   = tau
        self._y     = 0.0
        self._first = True

    # ------------------------------------------------------------------
    @staticmethod
    def _norm(a: float) -> float:
        """Wrap angle to [-180, 180] — O(1), no loop."""
        return (a + 180.0) % 360.0 - 180.0

    # ------------------------------------------------------------------
    def update(self, raw: float, dt: float | None = None) -> float:
        """
        Parameters
        ----------
        raw : float   New yaw measurement (degrees).
        dt  : float   Time since last call (seconds). Required if tau was set.
        """
        if self._first:
            self._y     = raw
            self._first = False
            return self._y

        # Compute effective alpha
        if self._tau is not None:
            if dt is None or dt <= 0.0:
                raise ValueError("dt must be a positive float when tau is set")
            alpha = 1.0 - math.exp(-dt / self._tau)
        else:
            alpha = self._alpha

        self._y += alpha * self._norm(raw - self._y)
        return self._y

    # ------------------------------------------------------------------
    def reset(self, value: float = 0.0) -> None:
        """Reinitialise le filtre (ex: après une perte de signal)."""
        self._y     = value
        self._first = True

    @property
    def value(self) -> float:
        """Dernière valeur filtrée."""
        return self._y