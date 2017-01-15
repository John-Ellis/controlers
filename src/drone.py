#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

class Drone:

    # Constructor. Tell us where the drone is and where it should initally go
    def __init__(self, dim=2, dur=100, time_delta=0.1):
        self.dim = dim
        self.dur = dur
        self.time_delta = time_delta
        self.thrust = np.zeros((self.dur, self.dim)) # m
        self.grd_pos = np.zeros((self.dur, self.dim)) # m
        self.grd_vel = np.zeros((self.dur, self.dim)) # m/s
        self.grd_acc = np.zeros((self.dur, self.dim)) # m/s^2
        self.air_vel = np.zeros((self.dur, self.dim)) # m/s
        self.air_acc = np.zeros((self.dur, self.dim)) # m/s^2
        self.wind_vel = np.zeros((self.dur, self.dim)) # m/s
        self.mass = np.ones(self.dur) # kg
        self.step = 0 #
        self.wind_vel[0] = 1.0 * (np.random.rand(self.dim) - 0.5)
        
    # Update wind gradually when it's called
    def updateWind(self, scale=4.0):
        self.wind_vel[self.step, :] = (self.wind_vel[self.step - 1] +
                                       scale *
                                       (np.random.rand(self.dim) - 0.5) *
                                       self.time_delta)

    # Update gas utilization # TODO use thrust
    def updateMass(self, thrust, scale=0.008):
        self.mass[self.step] = self.mass[self.step - 1] - scale * self.time_delta

    # Getters with noise
    def getGrdPos(self):
        return self.grd_pos[self.step] + np.random.normal(0, 0.5, self.dim)
    def getAirVel(self):
        return self.air_vel[self.step] + np.random.normal(0, 0.1, self.dim)
    def getGrdAcc(self):
        return self.grd_acc[self.step] + np.random.normal(0, 0.1, self.dim)
        
    # Apply a force from the controller
    def update(self, thrust):

        # Exit with false if we're done
        if self.step >= self.dur - 1 or thrust is None:
            return False

        # Update timestep
        self.step += 1
        s = self.step
        
        # Update state of drone
        self.thrust[s, :] = thrust
        self.updateWind()
        self.air_acc[s, :] = self.thrust[s] / self.mass[s - 1]
        self.air_vel[s, :] = self.air_vel[s - 1] + self.air_acc[s] * self.time_delta
        self.grd_vel[s, :] = self.air_vel[s] + self.wind_vel[s]
        self.grd_pos[s, :] = self.grd_pos[s - 1] + self.grd_vel[s] * self.time_delta
        self.grd_acc[s, :] = (self.grd_vel[s] - self.grd_vel[s - 1]) / self.time_delta
        self.updateMass(thrust)
        
        return True
    
    def plot(self):

        cmap = cm.jet
        c = np.linspace(0, 10, self.dur)
        
        fig = plt.figure(figsize=(10, 10))
        plt.title("Ground Position - Drone")
        plt.plot(self.grd_pos[:, 0], self.grd_pos[:, 1], zorder=1, color='black')
        plt.scatter(self.grd_pos[:, 0], self.grd_pos[:, 1], marker='o',
                    c=c, cmap=cmap, linewidth='0', zorder=2)
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.savefig('grd_pos_drone.png', bbox_inches='tight', dpi=160)

        fig = plt.figure(figsize=(10, 10))
        plt.title("Wind Velocity - Drone")
        plt.plot(self.wind_vel[:, 0], self.wind_vel[:, 1], zorder=1, color='black')
        plt.scatter(self.wind_vel[:, 0], self.wind_vel[:, 1], marker='o',
                    c=c, cmap=cmap, linewidth='0', zorder=2)
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.savefig('wind_vel_drone.png', bbox_inches='tight', dpi=160)
        
        fig = plt.figure(figsize=(10, 10))
        plt.title("Mass - Drone")
        plt.semilogy(self.mass, color='black')
        plt.ylim([1E-2, 10])
        plt.savefig('mass_drone.png', bbox_inches='tight', dpi=160)
