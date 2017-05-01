#!/usr/bin/env python3

import numpy as np
from drone import Drone
from controller import DeadController, AdaController, PIDposController

def main():
    np.random.seed(4)
    dst = np.array([0, 0])
    d = Drone(dur=1000)
    c = PIDposController(dst, dur=1000)
    done = False
    while not done:
        error = np.sqrt(np.sum(np.power(d.grd_pos[d.step] - dst, 2)))
        thrust = c.update(d.getGrdPos(), d.getAirVel(), d.getGrdAcc(), dst, error)
        if not d.update(thrust):
            done = True
    print(c.reportError())
    d.plot()
    c.plot()    

if __name__ == "__main__":
    main()
