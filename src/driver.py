#!/usr/bin/env python3

import numpy as np
from drone import Drone

def main():
    goal = np.array([5, 5])
    d = Drone(goal)
    while d.update(np.zeros(2)):
        if d.step % 100 == 0:
            print(d.step)
    d.plot()

if __name__ == "__main__":
    main()
