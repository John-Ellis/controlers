#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

class DeadController:

    # Constructor. Keep the history of controller
    def __init__(self, goal, dim=2, dur=100, time_delta=0.1):

        self.step = 0 # step
        self.dim = dim
        self.dur = dur
        self.time_delta = time_delta
        self.thrust_limit = 5.0
        self.vel_limit = 10.0

        # Measurements of drone
        self.dst = np.zeros((self.dur, self.dim)) # m
        self.grd_pos = np.zeros((self.dur, self.dim)) # m
        self.air_vel = np.zeros((self.dur, self.dim)) # m/s
        self.grd_acc = np.zeros((self.dur, self.dim)) # m/s^2
        self.thrust = np.zeros((self.dur, self.dim)) # N
        self.error = np.zeros((self.dur))
        self.dst[self.step, :] = goal
        
    # Use measurements to decide thrust
    def update(self, grd_pos, air_vel, grd_acc, dst, error):

        # Exit with None if we can't do any more
        if self.step >= self.dur - 1:
            return None

        # Update step and recorded measurements
        self.step += 1        
        self.grd_pos[self.step, :] = grd_pos
        self.air_vel[self.step, :] = air_vel
        self.grd_acc[self.step, :] = grd_acc
        self.dst[self.step, :] = dst
        self.error[self.step] = error

        self.updateThrust()
        
        return self.thrust[self.step]

    # Report error
    def reportError(self):
        return np.mean(self.error)

    def capThrust(self):
        magnitude = np.linalg.norm(self.thrust[self.step])
        if magnitude > self.thrust_limit:
            self.thrust[self.step] /= magnitude / self.thrust_limit
    
    # Update thrust
    def updateThrust(self):
        pass
    
    # Plot the history
    def plot(self):
        cmap = cm.jet
        c = np.linspace(0, 10, self.dur)

        fig = plt.figure(figsize=(10, 10))
        plt.title("Ground Position - Controller")
        plt.plot(self.grd_pos[:, 0], self.grd_pos[:, 1], zorder=1, color='black')
        plt.scatter(self.grd_pos[:, 0], self.grd_pos[:, 1], marker='o',
                    c=c, cmap=cmap, linewidth='0', zorder=2, label='Position')
        plt.scatter(self.dst[:, 0], self.dst[:, 1], marker='s',
                    c=c, cmap=cmap, linewidth='0', zorder=3, label='Goal')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.legend()
        plt.savefig('grd_pos_controller.png', bbox_inches='tight', dpi=160)
        
        fig = plt.figure(figsize=(10, 10))
        plt.title("Thrust")
        plt.plot(self.thrust[:, 0], self.thrust[:, 1], zorder=1, color='black')
        plt.scatter(self.thrust[:, 0], self.thrust[:, 1], marker='o',
                    c=c, cmap=cmap, linewidth='0', zorder=2)
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])        
        plt.savefig('thrust.png', bbox_inches='tight', dpi=160)

        fig = plt.figure(figsize=(10, 10))
        plt.title("Error")
        plt.plot(self.error, color='black')
        plt.ylim([0, 10])
        plt.savefig('error.png', bbox_inches='tight', dpi=160)
        

class BasicController(DeadController):
    def __init__(self, goal, dim=2, dur=100, time_delta=0.1):
        super().__init__(goal, dim, dur, time_delta)
        self.mass = np.ones((self.dur)) # kg
        self.wind_vel = np.zeros((self.dur, self.dim)) # m/s
        
    def updateThrust(self):
        self.mass[self.step] = (0.9 * self.mass[self.step - 1] +
                                0.1 * np.linalg.norm(self.thrust[self.step - 1]) * self.time_delta /
                                np.linalg.norm(self.air_vel[self.step] -
                                               self.air_vel[self.step - 1] + 1E-6))
        self.wind_vel[self.step] = (0.9 * self.wind_vel[self.step - 1] +
                                    (0.1 * ((self.grd_pos[self.step] - self.grd_pos[self.step - 1])
                                            / self.time_delta - self.air_vel[self.step])))
        dist = self.dst[self.step] - self.grd_pos[self.step]
        grd_vel = self.air_vel[self.step] + self.wind_vel[self.step]
        # Go faster if we're too far away
        desired_grd_vel = ((dist / (np.linalg.norm(dist) + 1E-9)) *
                            np.sqrt(2 * self.thrust_limit * np.linalg.norm(dist) /
                                         self.mass[self.step]))
        
        desired_air_vel = desired_grd_vel - self.wind_vel[self.step]
        desired_thrust = (self.mass[self.step] *
                          (desired_air_vel - self.air_vel[self.step]) /
                           self.time_delta)
        if np.sum(np.isnan(desired_thrust)) == 0:
            self.thrust[self.step] = desired_thrust
            self.capThrust()

    def plot(self):
        super().plot()

        cmap = cm.jet
        c = np.linspace(0, 10, self.dur)

        fig = plt.figure(figsize=(10, 10))
        plt.title("Mass - Controller")
        plt.semilogy(self.mass, color='black')
        plt.ylim([1E-2, 10])
        plt.savefig('mass_controller.png', bbox_inches='tight', dpi=160)
        
        fig = plt.figure(figsize=(10, 10))
        plt.title("Wind Velocity - Controller")
        plt.plot(self.wind_vel[:, 0], self.wind_vel[:, 1], zorder=1, color='black')
        plt.scatter(self.wind_vel[:, 0], self.wind_vel[:, 1], marker='o',
                    c=c, cmap=cmap, linewidth='0', zorder=2)
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])        
        plt.savefig('wind_vel_controller.png', bbox_inches='tight', dpi=160)        
        
        
class CMAController(DeadController):
    def __init__(self, goal, weights, dim=2, dur=100, time_delta=1.0):
        super().__init__(goal, dim, dur, time_delta)
        self.weights = weights

    def updateThrust(self):
        pass

    
