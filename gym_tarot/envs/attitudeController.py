# Attitude Controller Class
# Inherits from the controller abc.
# Implements a simple pid controller for the torque of an octorotor

import numpy as np 
from .controller import Controller

class AttitudeController(Controller):
    # Takes in as parameters:
    # j matrix - the moment of inertia matrix for the octorotor
    # kd - PID derivative gain
    # kp - PID proportional gain
    def __init__(self, ControllerArgs):
        self.kdphi = ControllerArgs["kdphi"]
        self.kpphi = ControllerArgs["kpphi"]
        self.kdpsi = ControllerArgs["kdpsi"]
        self.kppsi = ControllerArgs["kppsi"]
        self.kdtheta = ControllerArgs["kdtheta"]
        self.kptheta = ControllerArgs["kptheta"]


    # Outputs the torque for the octorotor
    # The current State objects is the Ocotorotrs current state consisting of Pos, Vel, Angle, AngleVel in 3 different directions each. 
    # The targetValue is the desired angularVelocity
    def output(self, currentState, targetValue):
        roll = -1*(currentState[6]*self.kdphi+self.kpphi*(currentState[9]-targetValue[0]))
        pitch = -1*(currentState[7]*self.kdtheta+self.kptheta*(currentState[10]-targetValue[1]))
        yaw = -1*(currentState[8]*self.kdpsi+self.kppsi*(currentState[11]-targetValue[2]))
        return roll, pitch, yaw
