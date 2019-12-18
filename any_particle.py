import numpy as np
import random
from real_construct_xy import *
import matplotlib.pyplot as plt

def emittance_per_particle(x, xp, reserved):
    sigma_x = weighted_avg_and_var(x * 10, x * 10)[2]
    sigma_xp = weighted_avg_and_var(xp * 1000, xp * 1000)[2]
    sigma_x_xp = weighted_avg_and_var(x * 10, xp * 1000)[2]
    emit_x = (sigma_x * sigma_xp - sigma_x_xp ** 2) ** 0.5
    alpha_x = -sigma_x_xp / emit_x
    beta_x = sigma_x / emit_x
    gamma_x = sigma_xp / emit_x
    emittances_x = gamma_x * (x * 10) ** 2 + 2 * alpha_x * x * 10 * xp * 1000 + beta_x *(xp * 1000) ** 2
    data = np.zeros((dis.shape[0], 3))
    data[:, 0] = x
    data[:, 1] = xp
    data[:, 2] = emittances_x
    data = data[data[:, 2].argsort()]
    data = data[:reserved, :]
    weights = np.ones(reserved)
    x, xp = part_regen(data[:, 0], data[:, 1], weights, 256, particle_num, 0)
    x = x - np.average(x)
    return x, xp
    
w = 6.0839
mc2 = 3751.13
gamma = w / mc2 + 1
btgm = (gamma ** 2 - 1) ** 0.5

percent = 0
I = 0.2315
dis_file = 'he2distribbuka.dst'
particle_num = 100000
dis = readDis(dis_file)
reserved = int(dis.shape[0] * (1 - percent))
x, xp = emittance_per_particle(dis[:, 0], dis[:, 1], reserved)
y, yp = emittance_per_particle(dis[:, 2], dis[:, 3], reserved)
#sample = random.sample(range(dis.shape[0]), particle_num)
#sample_dis = dis[sample, :]
#z, zp = sample_dis[:, 4], sample_dis[:, 5]
z, zp = emittance_per_particle(dis[:, 4], dis[:, 5], reserved)
new_dis = np.zeros((particle_num, 6))
new_dis[:, 0] = x
new_dis[:, 1] = xp
new_dis[:, 2] = y
new_dis[:, 3] = yp
new_dis[:, 4] = z
new_dis[:, 5] = zp
generate_new_dis(new_dis, I)
