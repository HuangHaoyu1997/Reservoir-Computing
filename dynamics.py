'''
Implementations of neural dynamics model

'''

import numpy as np
import matplotlib.pyplot as plt
import time

class HR:
    '''
    dx/dt = y + φ(x) - z + I
    dy/dt = ψ(x) - y
    dz/dt = r[s(x - x_R) - z] 
    φ(x) = -ax^3 + bx^2
    ψ(x) = c - dx^2
    
    parameter: a, b, c, d, r, s, x_R, I
    generally, a=1, b=3, c=1, d=5, s=4, x_R=-8/5
    parameter r governs the time scale of the neural adaptation, is sth of the order of 1e-3
    input current I ranges in [-10, 10]
    the 3rd equation allows a great variety of dynamic behaviors of the membrane potential
    '''
    
    def __init__(self,
                 params:dict,
                 dt=0.01,
                 ) -> None:
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']
        self.r = params['r']
        self.s = params['s']
        self.xR = params['xR']
        self.dt = dt
        self.reset()
        
    def reset(self,):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.xs = [self.x]
        self.ys = [self.y]
        self.zs = [self.z]

    
    def step(self, I):
        a, b, c, d, xR, dt, x, y, z = self.a, self.b, self.c, self.d, self.xR, self.dt, self.x, self.y, self.z
        r, s = self.r, self.s
        self.x = x + dt * (y - a*x**3 + b*x**2 - z + I)
        self.y = y + dt * (c - d*x**2 - y)
        self.z = z + dt * r * (s * (x - xR) - z)
        
        self.xs.append(self.x)
        self.ys.append(self.y)
        self.zs.append(self.z)

if __name__ == '__main__':
    params = {
        'a': 1., 
        'b': 3., 
        'c': 1., 
        'd': 5., 
        's': 4., 
        'r': 0.001,
        'xR': -8/5,
    }
    T = 1000
    model = HR(params, dt=0.1)
    for i in range(T):
        model.step(I=2 if i<500 else 0) # np.random.uniform(0,1)
    
    plt.plot(model.xs)
    plt.plot(model.xs[np.array(model.xs)>2.3],'.')
    plt.grid()
    plt.savefig('HR.png')
    print((np.array(model.xs)>2.3).sum())
