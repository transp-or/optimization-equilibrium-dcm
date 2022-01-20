import numpy as np
from scipy.optimize import fsolve
from scipy.special import lambertw

w = lambertw(1)
print(w)
print(w * np.exp(w))

alpha1 = 5
alpha2 = 4
beta = 0.1
cost1 = 0
cost2 = 0

def func(p):
    return [p[0] - (1 + lambertw(np.exp(alpha1 - 1 - cost1*beta)/(1 + np.exp(alpha2 - beta*p[1])))) / beta + cost1,
            p[1] - (1 + lambertw(np.exp(alpha2 - 1 - cost2*beta)/(1 + np.exp(alpha1 - beta*p[0])))) / beta + cost2]

optimalprices = fsolve(func, [0,0])
print(optimalprices)

p = [0, optimalprices[0], optimalprices[1]]
print(p)

prob = [0,0,0]
prob[0] = np.exp(0) / (np.exp(0) + np.exp(alpha1 - beta*p[1]) + np.exp(alpha2 - beta*p[2]))
prob[1] = np.exp(alpha1 - beta*p[1]) / (np.exp(0) + np.exp(alpha1 - beta*p[1]) + np.exp(alpha2 - beta*p[2]))
prob[2] = np.exp(alpha2 - beta*p[2]) / (np.exp(0) + np.exp(alpha1 - beta*p[1]) + np.exp(alpha2 - beta*p[2]))

print(prob)

print(np.multiply(prob,p))
