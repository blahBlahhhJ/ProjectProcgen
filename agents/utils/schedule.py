import numpy as np

class LinearSchedule():
    def __init__(self, start, end, n_steps):
        self.c = start
        self.c_start = start
        self.c_end = end
        self.n_steps = n_steps

    def step(self, t):
        if t <= self.n_steps:
            self.c = self.c_start - (self.c_start - self.c_end) / self.n_steps * t
        else:
            self.c = self.c_end
        return self.c

if __name__ == '__main__':
    test = LinearSchedule(5, 1, 4)
    for i in range(8):
        print(test.step(i))