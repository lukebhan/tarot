# Defines a simple forward euler integrator

class forwardEulerIntegrator:
    def __init__(self, initialVal, gain, sampleTime):
        self.initialVal = initialVal
        self.state = initialVal
        self.gain = gain
        self.sampleTime = sampleTime

    # U is derivative
    # x(n+1) = x(n) + u*k*T
    def step(self, u):
        self.state += self.gain*self.sampleTime*u
        return self.state

    def reset(self):
        self.state = self.initialVal
