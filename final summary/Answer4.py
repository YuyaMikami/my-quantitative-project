import numpy as np
import matplotlib.pyplot as plt

l = np.array([0.8027, 1.0, 1.2457])
NL = 3
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])
mu_1 = np.array([1.0/NL, 1.0/NL, 1.0/NL])
r = 1.025**20 - 1.0
tax = 0.30
av_i = 0.0
for i_y in range(NL):
    for i_m in range(NL):
        av_i += mu_1[i_y] * prob[i_y, i_m] * l[i_m]
ttax = av_i * tax
ppp = ttax * (1 + r)  # 一人当たり年金給付額
def util(cons, gamma):
    return max(cons, 1e-4)**(1.0-gamma)/(1.0-gamma)

gamma = 2.0
beta = 0.985**20
# グリッド設定
a_l = 0.0
a_u = 3.0
NA = 100
a = np.linspace(a_l, a_u, NA)

JJ = 3

v_p = np.zeros((JJ, NA, NL))
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)

iaplus_p = np.zeros((JJ, NA, NL), dtype=int)
aplus_p = np.zeros((JJ, NA, NL))
aplus = np.zeros((JJ, NA, NL))

# period 3
for ia in range(NA):
    for il in range(NL):
        cons = (1.0 + r) * a[ia]
        v[2, ia, il] = util(cons, gamma)

# period 2
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            cons = l[il] + (1.0 + r) * a[ia] - a[iap]
            if cons <= 0:
                reward[iap] = -np.inf
                continue
            EV = sum(prob[il, ilp] * v[2, iap, ilp] for ilp in range(NL))
            reward[iap] = util(cons, gamma) + beta * EV
        iaplus[1, ia, il] = np.argmax(reward)
        aplus[1, ia, il] = a[iaplus[1, ia, il]]
        v[1, ia, il] = reward[iaplus[1, ia, il]]

# period 1
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            cons = l[il] + (1.0 + r) * a[ia] - a[iap]
            if cons <= 0:
                reward[iap] = -np.inf
                continue
            EV = sum(prob[il, ilp] * v[1, iap, ilp] for ilp in range(NL))
            reward[iap] = util(cons, gamma) + beta * EV
        iaplus[0, ia, il] = np.argmax(reward)
        aplus[0, ia, il] = a[iaplus[0, ia, il]]
        v[0, ia, il] = reward[iaplus[0, ia, il]]

# period 3
for ia in range(NA):
    v_p[2, ia, :] = util((1.0+r)*a[ia] + ppp, gamma)

# period 2
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            cons = l[il]*(1 - tax) + (1.0+r)*a[ia] - a[iap]
            if cons <= 0:
                reward[iap] = -np.inf
                continue
            EV = sum(prob[il, ilp] * v_p[2, iap, ilp] for ilp in range(NL))
            reward[iap] = util(cons, gamma) + beta * EV
        iaplus_p[1, ia, il] = np.argmax(reward)
        aplus_p[1, ia, il] = a[iaplus_p[1, ia, il]]
        v_p[1, ia, il] = reward[iaplus_p[1, ia, il]]

# period 1（若年期）
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            cons = l[il] + (1.0+r)*a[ia] - a[iap]
            if cons <= 0:
                reward[iap] = -np.inf
                continue
            EV = sum(prob[il, ilp] * v_p[1, iap, ilp] for ilp in range(NL))
            reward[iap] = util(cons, gamma) + beta * EV
        iaplus_p[0, ia, il] = np.argmax(reward)
        aplus_p[0, ia, il] = a[iaplus_p[0, ia, il]]
        v_p[0, ia, il] = reward[iaplus_p[0, ia, il]]
av_util = 0.0
for il in range(NL):
  av_util += mu_1[il]*v[0, 0, il]

av_util_p = 0.0
for il in range(NL):
  av_util_p += mu_1[il]*v_p[0, 0, il]

print(f"年金なしの場合の経済全体の平均期待生涯効用: {av_util:.4f}")
print(f"年金ありの場合の経済全体の平均期待生涯効用: {av_util_p:.4f}")