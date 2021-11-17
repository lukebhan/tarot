# Provides the control allocation for 
# Octorotor model

import math
import numpy as np

class Allocator:
    def __init__(self, AllocatorParams):
        # Set parameters
        # Max speed
        self.maxSpeed = AllocatorParams["maxSpeed"]
        self.maxSpeedSquare = self.maxSpeed*self.maxSpeed
        # arm length 
        l = AllocatorParams["l"]
        # thrust motor constant [Ns^2/rad^2]
        b = AllocatorParams["b"]
        # rotor drag constant [Nms^2/rad^2]
        d = AllocatorParams["d"]
        # voltage saturation
        self.voltageSat = AllocatorParams["voltageSat"]

        # Angle Limits
        angSmall= AllocatorParams["minAng"]
        angLarge = AllocatorParams["maxAng"]

        # Initial allocation Matrix
        self.AF = np.array([
            [b, b, b, b, b, b, b, b],
            [b*l*angLarge, b*l*angSmall, -b*l*angSmall, -b*l*angLarge, -b*l*angLarge, -b*l*angSmall, b*l*angSmall, b*l*angLarge],
            [-b*l*angSmall, -b*l*angLarge, -b*l*angLarge, -b*l*angSmall, b*l*angSmall, b*l*angLarge , b*l*angLarge, b*l*angSmall],
            [d, -d, d, -d, d, -d, d, -d]])

        # Calculate Inverse
        self.AFinv = np.linalg.pinv(self.AF)

    def getRefVoltage(self, uDesired):
        # first constrain omega.
        omegaS = self.AFinv.dot(uDesired) 
        vref = np.zeros(8)
        for idx, val in enumerate(omegaS):
            if val < 0:
                vref[idx] = 0
            elif val > self.maxSpeedSquare:
                vref[idx]=self.maxSpeed
            else:
                vref[idx]=math.sqrt(val)
        for idx,val in enumerate(vref):
            if(val*self.voltageSat > 26.1):
                vref[idx] = 26.1
            else:
                vref[idx]=val*self.voltageSat
        return vref
