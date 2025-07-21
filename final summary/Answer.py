import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def util(cons, gamma):
    return max(cons, 1e-4)**(1.0-gamma)/(1.0-gamma)

gamma = 2.0
beta = 0.985**20
r = 1.025**20-1.0
JJ = 3
l = np.array([0.8027, 1.0, 1.2457])
NL = 3
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])
mu_1 = np.array([1.0/NL, 1.0/NL, 1.0/NL])
mu_2 = np.zeros(NL)

for il in range(NL):
    for ilp in range(NL):
        mu_2[ilp] += prob[il, ilp]*mu_1[il]


# grids
a_l = 0.0
a_u = 3.0
NA = 100
a = np.linspace(a_l, a_u, NA)

# initialization
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)
aplus = np.zeros((JJ, NA, NL))



# period 3
for ia in range(NA):
    v[2, ia, :] = util((1.0+r)*a[ia], gamma)


# period 2
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):

          EV = 0.0
          for ilp in range(NL):
              EV += prob[il, ilp]*v[2, iap, ilp]

          reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*EV

        iaplus[1, ia, il] = np.argmax(reward)
        aplus[1, ia, il] = a[iaplus[1, ia, il]]
        v[1, ia, il] = reward[iaplus[1, ia, il]]

# period 1
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):

            EV = 0.0
            for ilp in range(NL):
                EV += prob[il, ilp]*v[1, iap, ilp]

            reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*EV

        iaplus[0, ia, il] = np.argmax(reward)
        aplus[0, ia, il] = a[iaplus[0, ia, il]]
        v[0, ia, il] = reward[iaplus[0, ia, il]]

plt.figure()
plt.plot(a, aplus[0, :, 0], label='Low')
plt.plot(a, aplus[0, :, 1], label='Mid')
plt.plot(a, aplus[0, :, 2], label='High')
plt.title("policy function")
plt.xlabel("Assets excluding interest at the beginning of the younger period")
plt.ylabel("Savings(Assets excluding interest for the next period)")
plt.ylim(a_l, a_u)
plt.grid(True)
plt.legend()
plt.show()
