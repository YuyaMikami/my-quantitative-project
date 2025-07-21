import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
l = np.array([0.8027, 1.0, 1.2457])
NL = 3
r = 1.025**20-1.0
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])

mu_1 = np.array([1.0/NL, 1.0/NL, 1.0/NL])

mu_2 = np.zeros(NL)
for il in range(NL):
    for ilp in range(NL):
        mu_2[ilp] += prob[il, ilp] * mu_1[il]

tax = 0.30
av_i = 0.0
for i_y in range(NL):
    for i_m in range(NL):
        av_i += \
            mu_1[i_y] * prob[i_y, i_m] * l[i_m]

ttax = av_i * tax

g_asset = ttax * (1 + r)
ppp = g_asset

print(f"中年期における政府の総税収: {ttax:.4f}")
print(f"一人当たりの年金額: {ppp:.4f}")