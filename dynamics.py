'''
Implementations of neural dynamics model

'''

import numpy as np
import matplotlib.pyplot as plt
import time

class Izhikevich:
    '''
    v' = 0.04v^2 + 5v + 140 - u + I
    u' = a(bv - u)
    '''
    def __init__(self) -> None:
        pass

def IzhikevichModel(v0, u0, dt, I, a, b):
    '''
    step function of Izhikevich model
    v' = 0.04v^2 + 5v + 140 - u + I
    u' = a(bv - u)
    '''
    v = v0 + dt * (0.04*v0**2 + 5*v0 + 140 - u0 + I)
    u = u0 + dt * a * (b * v0 - u0)
    return v, u

def IzhikevichSimulation(v0, u0, dt, a, b, c, d, T, thr):
    vs, us, spikes = [], [], []
    T = np.arange(0, T, dt)
    for t in T:
        I = 5
        v, u = IzhikevichModel(v0, u0, dt, I, a, b)
        if v>= thr:
            spikes.append(1)
            v = c
            u += d
        else: spikes.append(0)
        
        vs.append(v)
        us.append(u)
        v0 = v
        u0 = u
    return vs, us, spikes

class Hindmarsh_Rose:
    '''
    Hindmarsh-Rose model
    
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
    # params = {
    #     'a': 1., 
    #     'b': 3., 
    #     'c': 1., 
    #     'd': 5., 
    #     's': 4., 
    #     'r': 0.001,
    #     'xR': -8/5,
    # }
    # T = 30
    # model = Hindmarsh_Rose(params, dt=0.1)
    # for i in range(T):
    #     model.step(I=2 if i<20 else 0) # np.random.uniform(0,1)
    
    # plt.subplot(121)
    # plt.plot(model.xs)
    # # plt.plot(model.xs[np.array(model.xs)>2.3],'.')
    # plt.grid()
    
    # plt.subplot(122)
    # spikes = [1 if x >2.3 else 0 for x in model.xs]
    # plt.plot(spikes,'.')
    # print((np.array(model.xs)>2.3).sum())
    
    # plt.savefig('HR.png')
    
    
    
    vs, us, spikes = IzhikevichSimulation(v0=-65,
                                            u0=0.2*-65,
                                            dt=0.01,
                                            a=0.02,
                                            b=0.2,
                                            c=-65,
                                            d=2,
                                            T=1000,
                                            thr=30)
    T = np.arange(0, 1000, 0.01)
    plt.subplot(311)
    plt.plot(T, vs)
    plt.subplot(312)
    plt.plot(T, us)
    plt.subplot(313)
    plt.plot(T, spikes)
    
    plt.savefig('IZH.png')
    print(np.sum(np.array(spikes)==1))
