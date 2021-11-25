import math
import numpy as np
import scipy.integrate

class Airframe:
    def __init__(self, TarotParams, sampleTime):
        self.g = TarotParams["g"]
        self.m = TarotParams["m"]
        self.l = TarotParams["l"]
        self.b = TarotParams["b"]
        self.d = TarotParams["d"]
        self.angSmall = TarotParams["minAng"]
        self.angLarge = TarotParams["maxAng"]
        self.state = np.zeros(12, dtype = np.float32)
        self.state[2] = 5
        self.ode = scipy.integrate.ode(self.state_dot)
        self.Ixx = TarotParams["Ixx"]
        self.Iyy = TarotParams["Iyy"]
        self.Izz = TarotParams["Izz"]
        self.forces = np.zeros(3, dtype="float32")
        self.moments = np.zeros(3, dtype="float32")
        self.sampleTime = sampleTime
        self.rotationMatrix = np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]])
        self.rotationMatrixInv = np.linalg.pinv(self.rotationMatrix)
        self.integrator = scipy.integrate.ode(self.state_dot)

    def state_dot(self, time, state):
        state_dot = np.zeros(12, dtype="float32") 
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]
        accel = self.forces/self.m - np.cross(self.state[6:9], self.state[3:6])
        state_dot[3:6] = accel
        angularAccel = self.rotationMatrixInv.dot(self.moments-np.cross(self.state[6:9], self.rotationMatrix.dot(self.state[6:9])))
        state_dot[6:9] = angularAccel
        sinPhi = math.sin(self.state[9])
        cosPhi = math.cos(self.state[9])
        tanTheta = math.tan(self.state[10])
        cosTheta = math.cos(self.state[10])
        state_dot[9] = self.state[6] + sinPhi*tanTheta*self.state[7]+cosPhi*tanTheta*self.state[8]
        state_dot[10]= cosPhi*self.state[7]-sinPhi*self.state[8]
        state_dot[11] = sinPhi/cosTheta*self.state[7] + cosPhi/cosTheta*self.state[8]
        return state_dot

    def generateMoments(self, omega, phi, theta, psi):
        thrust = self.b*((2*math.pi)*omega/60)**2
        thrust_total = np.round(np.sum(thrust)*100)/100
        cosPsi = math.cos(psi)
        sinPsi = math.sin(psi)
        cosPhi = math.cos(phi)
        sinPhi = math.sin(phi)
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        # calculate forces
        localForceMatrix = np.array([[cosPsi*cosTheta, cosPsi*sinTheta*sinPhi-sinPsi*cosPhi, cosPsi*sinTheta*cosPhi+sinPsi*sinPhi], [sinPsi*cosTheta, sinPsi*sinTheta*sinPhi+cosPsi*cosPhi, sinPsi*sinTheta*cosPhi-cosPsi*sinPhi], [-sinTheta, cosTheta*sinPhi, cosTheta*cosPhi]])
        fTotal = np.dot(localForceMatrix.transpose(),np.array([0, 0, -self.m*self.g]))+np.array([0, 0, thrust_total])
        # calculate moments
        moment = self.d*((2*math.pi)*omega/60)**2
        T = thrust
        anglg=self.angLarge
        angsm=self.angSmall
        mx = self.l*(thrust[0]*self.angLarge + thrust[1]*self.angSmall - thrust[2]*self.angSmall - thrust[3]*self.angLarge - thrust[4]*self.angLarge - thrust[5]*self.angSmall + thrust[6]*self.angSmall + thrust[7]*self.angLarge)
        my = self.l*(-thrust[0]*self.angSmall-thrust[1]*self.angLarge-thrust[2]*self.angLarge-thrust[3]*self.angSmall+thrust[4]*self.angSmall+thrust[5]*self.angLarge + thrust[6]*self.angLarge + thrust[7]*self.angSmall)
        mz = moment[0]-moment[1]+moment[2]-moment[3]+moment[4]-moment[5]+moment[6]-moment[7]
        mArray = np.array([mx, my, mz])
        return fTotal, mArray

    def update(self, omega):
        self.forces, self.moments = self.generateMoments(omega, self.state[9], self.state[10], self.state[11])
        # use forward euler method for simplified integration 
        self.integrator.set_initial_value(self.state, 0)
        self.state= self.integrator.integrate(self.integrator.t+self.sampleTime)
        #for i in range(100):
            #self.state = self.integrator.integrate(self.integrator.t+self.sampleTime/100)
        return self.state

    def getState(self):
        return self.state

    def getRotMatrix(self):
        sPhi = math.sin(self.state[9])
        tTheta = math.tan(self.state[10])
        cPhi = math.cos(self.state[9])
        cTheta = math.cos(self.state[10])
        mat = np.array([1, sPhi*tTheta, cPhi*tTheta, 0, cPhi, -sPhi, 0, sPhi/cTheta, cPhi/cTheta])
        return mat
        
