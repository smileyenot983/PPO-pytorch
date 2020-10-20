import argparse
import os
import pickle

import gym
import numpy as np
from torch.autograd import Variable
# import main
import models

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pybulletgym
from running_state import ZFilter


#TODO:
# 1.load post evaluation results with corresponding policies
# 2.choose the best ones

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='Name of folder which contains results of experiments')

args = parser.parse_args()

#counting number of seeds used for experiments
n_seeds=0
for f in os.listdir(args.src):
# for f in os.listdir(src):
    if f.startswith('seed'):
        n_seeds+=1

print('n seeds = ' + str(n_seeds))



#it is assumed that every seed was run with same settings(same types of noise and same initial sigmas)
first_folder = os.listdir(args.src)[0]
all_settings = os.listdir(args.src + '/' +first_folder + '/post_evaluation')
# print(first_folder)
# print(all_settings)

post_noise_policies = {}
post_nonoise_policies = {}

for seed in range(1,n_seeds+1):
    data_src = args.src + '/seed' + str(seed) + '/post_evaluation/'

    seed_settings = os.listdir(data_src)

    for setting in seed_settings:
        with open(data_src + setting,'rb') as f:
            setting_results = pickle.load(f)

        if 'Nonoise' in setting:
            post_nonoise_policies[setting] = setting_results
        else:
            post_noise_policies[setting] = setting_results


#here are
post_noise_sorted = sorted(post_noise_policies,key=lambda x:x[1])
post_nonoise_sorted = sorted(post_nonoise_policies,key=lambda x:x[1])

# print(len(post_nonoise_policies))




'''statistical test'''
from scipy import stats

#number of best policies to compare
n_policies = 5
#here is Mann Whitney test which compares n best policies with and without noise
for i in range(n_policies):
    #rewards with parameter noise
    with_noise = post_noise_policies[post_noise_sorted[i]]

    without_noise = post_nonoise_policies[post_nonoise_sorted[i]]

    # Mann-WHitney U test:
    stat, p = stats.mannwhitneyu(with_noise, without_noise)

    print("p_value")
    print(p)
    # print(stat)


# best_noise = post_noise_sorted[-1:-5]
#
# best_nonoise = post_nonoise_sorted[-1:-5]


# print('______________________best noise__________________-')
# print(best_noise)
# print('______________________best without noise__________________-')
# print(best_nonoise)

##plotting 2 best policies
# t1 = np.arange(len(post_noise_policies[best_noise]))
# rewards_noise = post_noise_policies[best_noise]
#
# t2 = np.arange(len(post_nonoise_policies[best_nonoise]))
# rewards_nonoise = post_nonoise_policies[best_nonoise]
#
# print(rewards_noise)
#
# plt.plot(t1,rewards_noise, label=best_noise)
# plt.plot(t2,rewards_nonoise, label=best_nonoise)
# plt.legend()
# plt.show()