import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='Name of folder which contains results of experiments(2 folders with data and plots)')

args = parser.parse_args()

# src = './CartPoleSwingUp-v3_tests'

n_seeds = len(os.listdir(args.src))

def load_data(data_src,data_dict,seed):
    '''function for loading data obtained from training'''

    #getting all files names in given subdirectory
    settings = os.listdir(data_src)

    for setting in settings:


        if 's' + str(seed) in setting:
            new_setting = setting.replace('s' + str(seed),'')

        # print(setting)
        # print(new_setting)

        if new_setting not in data_dict:
            data_dict[new_setting] = []

        # if settings_data[setting] not in settings:
        #     settings_data
        with open(data_src + '/' + setting,'rb') as f:
            data_dict[new_setting].append(pickle.load(f))

# dict for saving all data from experiments
all_variations = {}

#iterating through all seeds
for seed in range(1,n_seeds+1):
    #changing directory to given seed and opening folder which contains data
    data_src = args.src + '/seed' + str(seed) + '/data'

    load_data(data_src,all_variations,seed)
    # os.chdir(src + '/seed' + str(1) + '/data')

variations_averages = {}

#key - different variations of noise
for key in all_variations.keys():

    variations_averages[key] = []

    #iterating through all experiments(300 in my case)
    for n_exp in range(len(all_variations[key][0])):
        exp_average = 0.0
        #taking values from all seeds
        for seed in range(n_seeds):
        #summing experiment rewards from all seeds
            exp_average += all_variations[key][seed][n_exp]

        #taking the average for experiment
        exp_average /= n_seeds

        variations_averages[key].append(exp_average)

##plotting
key1 = list(variations_averages.keys())[0]
t = np.arange(len(variations_averages[key1]))
# print(t)

plt.rcParams["figure.figsize"] = (10,5)

for key in variations_averages.keys():
    plot_name = key
    plt.plot(t,variations_averages[key],label=key)
    plt.legend()
plt.show()
