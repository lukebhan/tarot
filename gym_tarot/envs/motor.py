# A DC Motor 

from .eulerForward import forwardEulerIntegrator
import numpy as np
import math

class Motor:
    def __init__(self, MotorParams, initCond, sampleTime):
        # Initialize Parameters
        self.torqueConst = MotorParams["torqueConst"]
        self.equivResistance = MotorParams["equivResistance"]
        self.currentSat = MotorParams["currentSat"]
        self.staticFric = MotorParams["staticFric"]
        self.damping = MotorParams["damping"]
        self.J = MotorParams["J"]
        self.thrustCoeff = MotorParams["thrustCoeff"]

        # We need to define the integrator for our mechanical method
        self.integrator = forwardEulerIntegrator(initCond, 1, sampleTime)
        # We start with an angularSpeed and torque of 0.
        self.angularSpeed = 365
        self.torque = 0
        self.setBroken = False

    def electrical(self, voltage):
        # Calculate current from voltage and angular speed
        current = (1/self.equivResistance)*(voltage - self.angularSpeed*self.torqueConst)
        current = min(max(current, 0), self.currentSat)
        # We assume kt=ke for an ideal square wave motor
        eTorque = current*self.torqueConst
        return eTorque, current

    def mechanical(self, eTorque):
        # convert to angular speed
        res = eTorque-self.staticFric-self.torque-(self.angularSpeed*self.damping)
        res *= (1/self.J)
        self.angularSpeed = self.integrator.step(res)
        # convert to torque and thrusts
        rpm= self.angularSpeed*(60/(2*math.pi))
        # we use the tarot propeller model here to calculate thrust and torque
        #thrust = self.thrustCoeff *1.225*(rpm/60)*(rmp/60)*.47**4
        self.torque = (2.138e-8)*rpm*rpm+(-1.279e-5)*rpm
        return rpm

    def getAngularSpeed(self, voltage):
        if self.setBroken:
            return 0
        eTorque, current = self.electrical(voltage)
        rpm = self.mechanical(eTorque)
        return rpm

    def setRes(self, res):
        self.equivResistance = res

    def broken(self):
        self.setBroken = True
