class ConvergenceTester:
    def __init__(self, tol: float = 1e-3, tol_steps=20):
        self.tol = tol
        self.tol_steps = tol_steps

        self.prev_value = None
        self.current_value = None
        self.unconverged_steps = 0

    def add_value(self, value):
        self.prev_value = self.current_value
        self.current_value = value

    def has_converged(self):
        if self.prev_value is None or self.current_value is None:
            return False

        if abs(self.current_value - self.prev_value) <= self.tol:
            self.unconverged_steps += 1
        else:
            self.unconverged_steps = 0

        if self.unconverged_steps >= self.tol_steps:
            return True
        
        return False