class PID_CONTROLLER:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def update(self, set_point, current_value, dt):
        error = set_point - current_value
        self.integral += error * dt
        derivate = (error  - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivate
        self.previous_error = error
        return output
