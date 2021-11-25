import numpy as np
import math
import pandas as pd
import gym
from gym import spaces
import torch
from .autoencoder import AE
from .motor import Motor
from .airframe import Airframe
from .allocation import Allocator
from .altitudeController import AltitudeController
from .positionController import PositionController
from .attitudeController import AttitudeController

class TarotBaseEnv(gym.Env):
    def __init__(self):
        super(TarotBaseEnv, self).__init__()
        self.path = "../../gym_tarot/envs/"
        # Set spaces
        self.observation_space = spaces.Box(np.full(24, -np.inf, dtype="float32"), np.full(24, np.inf, dtype="float32"), dtype="float32")
        self.action_space = spaces.Box(np.full(4, -10, dtype="float32"), np.full(4, 10, dtype="float32"))

        # Create model
        # Initialize parameters
        self.psiRef = 0
        MotorParams = {"torqueConst": 0.0265, "equivResistance": 0.2700, "currentSat": 38, "staticFric": 0, "damping": 0, "J": 5.0e-5, "thrustCoeff": 0.065}
        icMotor = 0
        sampleTime = 0.01
        self.motorArray = []
        for i in range(8):
            m = Motor(MotorParams, icMotor, sampleTime)
            self.motorArray.append(m)

        # Initialize Airframe
        TarotParams = {"g": 9.80, "m": 10.66, "l": 0.6350, "b": 9.8419e-05, "d": 1.8503e-06, "minAng": math.cos(math.pi/8), "maxAng": math.cos(3*math.pi/8), "Ixx": 0.2506, "Iyy": 0.2506, "Izz": 0.4538, "maxSpeed": 670, "voltageSat": 0.0325}
        self.tarot = Airframe(TarotParams, sampleTime)

        # Initialize Control Allocation
        self.allocator = Allocator(TarotParams)

        # Initial controllers
        AltitudeControllerParams = {'m': 10.66, 'g':9.8, 'kdz': -1, 'kpz': -0.5}
        self.altitudeController = AltitudeController(AltitudeControllerParams)
        PositionControllerParams = { 'kpx': 0.1, 'kdx': 0, 'kpy': 0.1, 'kdy': 0, 'min_angle': -12*math.pi/180, 'max_angle': 12*math.pi/180 }
        self.positionController = PositionController(PositionControllerParams)
        AttitudeControllerParams = {"kdphi": 1, "kpphi": 3, "kdpsi": 1, "kppsi": 3, "kdtheta": 1, "kptheta": 3}
        self.attitudeController = AttitudeController(AttitudeControllerParams)

        # Create autoencoder
        self.ae = AE()
        self.ae.load_state_dict(torch.load(self.path+'ae4.pt'))
        self.ae.eval()

    def step(self, action):
        # update trajectories
        error = math.sqrt((self.xref-self.state[0])*(self.xref-self.state[0]) + (self.yref-self.state[1]) * (self.yref-self.state[1])+ (self.zref-self.state[2]) * (self.zref-self.state[2]))
        if error < 0.5:
            self.xref = self.xrefarr[self.stepCount]
            self.yref = self.yrefarr[self.stepCount]
            self.zref = 5
            self.stepCount += 1
        self.curTime += self.sampleTime

        fz = self.altitudeController.output(self.state, self.zref)
        thetaRef, phiRef = self.positionController.output(self.state, [self.xref, self.yref])
        roll, pitch, yaw = self.attitudeController.output(self.state, [phiRef, thetaRef, self.psiRef]) 
        # 1.1 / 1 and .9 is to -1
        uDesired = [fz+action[0], roll+action[1], pitch+action[2], yaw+action[3]]
        refVoltage = self.allocator.getRefVoltage(uDesired)
        rpm = np.zeros(8, dtype=np.float32)
        for idx, motor in enumerate(self.motorArray):
            rpm[idx] = motor.getAngularSpeed(refVoltage[idx])
        self.state = self.tarot.update(rpm)
        # Add noise
        if self.stepCount % 100 == 0:
            self.state[0] += np.random.normal(0, 0.1)
            self.state[1] += np.random.normal(0, 0.1)

        # Build next obs space
        matrix = self.tarot.getRotMatrix()
        err = self.state[0:3] - np.array([self.xref, self.yref, self.zref])
        obsFullDim = np.concatenate([self.state, matrix, err])
        #voltageArr = np.zeros(8)
        #obsSpace = np.concatenate([self.ae.encode(torch.from_numpy(obsFullDim).float()).detach().numpy(), voltageArr])

        # Check terminate condition
        finish = self.terminate()
        error = math.sqrt((self.xref-self.state[0])*(self.xref-self.state[0]) + (self.yref-self.state[1]) * (self.yref-self.state[1]) + (self.zref-self.state[2]) * (self.zref-self.state[2]))
        reward = (-error+3)/3
        return obsFullDim, reward, finish, {"traj": self.traj, "xref": self.xref, "yref": self.yref, "x": self.state[0], "y": self.state[1]}

    def reset(self):
        # Recreate Model
        MotorParams = {"torqueConst": 0.0265, "equivResistance": 0.2700, "currentSat": 38, "staticFric": 0, "damping": 0, "J": 5.0e-5, "thrustCoeff": 0.065}
        icMotor = 0
        self.sampleTime = 0.01
        self.motorArray = []
        for i in range(8):
            icMotor = 0
            m = Motor(MotorParams, icMotor, self.sampleTime)
            self.motorArray.append(m)
        self.motorArray[3].setRes(0.29)

        # Initialize Airframe
        TarotParams = {"g": 9.80, "m": 10.66, "l": 0.6350, "b": 9.8419e-05, "d": 1.8503e-06, "minAng": math.cos(math.pi/8), "maxAng": math.cos(3*math.pi/8), "Ixx": 0.2506, "Iyy": 0.2506, "Izz": 0.4538, "maxSpeed": 670, "voltageSat": 0.0325}
        self.tarot = Airframe(TarotParams, self.sampleTime)

        # Initialize Control Allocation
        self.allocator = Allocator(TarotParams)

        # Initial controllers
        AltitudeControllerParams = {'m': 10.66, 'g':9.8, 'kdz': -1, 'kpz': -0.5}
        self.altitudeController = AltitudeController(AltitudeControllerParams)
        PositionControllerParams = { 'kpx': 0.1, 'kdx': 0, 'kpy': 0.1, 'kdy': 0, 'min_angle': -12*math.pi/180, 'max_angle': 12*math.pi/180 }
        self.positionController = PositionController(PositionControllerParams)
        AttitudeControllerParams = {"kdphi": 1, "kpphi": 3, "kdpsi": 1, "kppsi": 3, "kdtheta": 1, "kptheta": 3}
        self.attitudeController = AttitudeController(AttitudeControllerParams)

        # Setup trajectory and finish condition 
        self.traj = np.random.randint(0, 3)
        self.traj = 0
        if self.traj == 0:
            self.xrefarr = pd.read_csv(self.path+"xref8traj.csv", header=None).iloc[:, 1]
            self.yrefarr = pd.read_csv(self.path+"yref8traj.csv", header=None).iloc[:, 1]
        elif self.traj == 1:
            self.xrefarr = pd.read_csv(self.path+"xrefEtraj.csv", header=None).iloc[:, 1]
            self.yrefarr = pd.read_csv(self.path+"yrefEtraj.csv", header=None).iloc[:, 1]
        elif self.traj == 2:
            self.yrefarr = pd.read_csv(self.path+"yrefZigtraj.csv", header=None).iloc[:, 1]
            self.xrefarr = pd.read_csv(self.path+"xrefZigtraj.csv", header=None).iloc[:, 1]
        self.endTime = len(self.yrefarr)
        self.curTime = 0
        self.xref = self.xrefarr[self.curTime]
        self.yref = self.yrefarr[self.curTime]
        self.zref = 5
        self.stepCount = 0

        self.state = self.tarot.getState()
        # build obs space
        matrix = self.tarot.getRotMatrix()
        err = self.state[0:3] - np.array([self.xref, self.yref, self.zref])
        obsFullDim = np.concatenate([self.state, matrix, err])
        voltageArr = np.zeros(8)
        #obsSpace = np.concatenate([self.ae.encode(torch.from_numpy(obsFullDim).float()).detach().numpy(), voltageArr])
        obsSpace = obsFullDim
        return obsSpace

    def terminate(self):
        error = math.sqrt((self.xref-self.state[0])*(self.xref-self.state[0]) + (self.yref-self.state[1]) * (self.yref-self.state[1]) + (self.zref-self.state[2]) * (self.zref-self.state[2]))
        return self.stepCount == len(self.xrefarr)-1  or error > 3 or np.isclose(self.curTime, self.endTime*1.5, 1e-9)
