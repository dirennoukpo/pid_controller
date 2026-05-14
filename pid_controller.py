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
