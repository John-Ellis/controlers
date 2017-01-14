#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class Drone:

    # Constructor. Tell us where the drone is and where it should initally go
    def __init__(self, dst, time_delta=1.0):
        self.dim = 2
        self.dur = 10000
        self.dst = np.zeros((self.dur, self.dim)) # m
        self.thrust = np.zeros((self.dur, self.dim)) # m
        self.grd_pos = np.zeros((self.dur, self.dim)) # m
        self.grd_vel = np.zeros((self.dur, self.dim)) # m/s
        self.grd_acc = np.zeros((self.dur, self.dim)) # m/s^2
        self.air_vel = np.zeros((self.dur, self.dim)) # m/s
        self.air_acc = np.zeros((self.dur, self.dim)) # m/s^2
        self.wind_vel = np.zeros((self.dur, self.dim)) # m/s
        self.mass = np.ones(self.dur) # kg
        self.step = 1 # s

        # Update based on variables
        self.time_delta = time_delta
        self.dst[self.step] = dst

    # Update wind gradually when it's called
    def updateWind(self, scale=1.0):
        self.wind_vel[self.step] = (self.wind_vel[self.step - 1] +
                                    scale *
                                    (np.random.rand(self.dim) - 0.5) *
                                    self.time_delta)

    # Update gas utilization # TODO use thrust
    def updateMass(self, thrust, scale=0.001):
        self.mass[self.step] = self.mass[self.step - 1] - scale * self.time_delta
        
    # Apply a force from the controller
    def update(self, thrust):

        # Exit with false if we're done
        if self.step >= self.dur - 1:
            return False

        # Update timestep
        self.step += 1
        s = self.step
        
        # Update state of drone
        self.thrust[s] = thrust
        self.updateWind()
        self.air_acc[s] = self.thrust[s] / self.mass[s - 1]
        self.air_vel[s] = self.air_vel[s - 1] + self.air_acc[s] * self.time_delta
        self.grd_vel[s] = self.air_vel[s] + self.wind_vel[s]
        self.grd_pos[s] = self.grd_pos[s - 1] + self.grd_vel[s] * self.time_delta
        self.grd_acc[s] = (self.grd_vel[s] - self.grd_vel[s - 1]) / self.time_delta
        self.updateMass(thrust)
        
        return True

    def plot(self):
        fig = plt.figure(figsize=(10, 10))
        plt.title("Ground Position")
        plt.plot(self.grd_pos[:, 0], self.grd_pos[:, 1], '-')
        plt.savefig('grd_pos.png', bbox_inches='tight', dpi=160)

        fig = plt.figure(figsize=(10, 10))
        plt.title("Wind Velocity")
        plt.plot(self.wind_vel[:, 0], self.wind_vel[:, 1], '-')
        plt.savefig('wind_vel.png', bbox_inches='tight', dpi=160)
        
