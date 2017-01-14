#!/usr/bin/env python3

import numpy as np
from drone import Drone
from controller import Controller

def main():
    dst = np.zeros(2)
    d = Drone()
    c = Controller(dst)
    done = False
    while not done:
        error = np.sqrt(np.sum(np.power(d.grd_pos[d.step] - dst, 2)))
        thrust = c.update(d.getGrdPos(), d.getAirVel(), d.getGrdAcc(), dst, error)
        if not d.update(thrust):
            done = True
        
    d.plot()
    c.plot()    

if __name__ == "__main__":
    main()
