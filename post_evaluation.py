'''this script looks all data generated during experiments, choose n best policies and post evaluates them all to choose the best one'''
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

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='Name of folder which contains results of experiments(2 folders with data and plots)')
parser.add_argument('--env', type=str, help='environment name')

args = parser.parse_args()
#counting number of seeds used for experiments
n_seeds=0
for f in os.listdir(args.src):
# for f in os.listdir(src):
    if f.startswith('seed'):
        n_seeds+=1

print('n seeds = ' + str(n_seeds))


'''Here are modified policy networks, to be able to use them at post evaluation'''


class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)

        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.module_list_current = [self.affine1, self.affine2, self.action_mean, self.action_log_std]

        self.module_list_old = [None] * len(
            self.module_list_current)  # self.affine1_old, self.affine2_old, self.action_mean_old, self.action_log_std_old]



    def forward(self, x, old=False):

        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))

            action_mean = self.module_list_old[2](x)
            # action_mean = action_mean.reshape(1, action_mean.shape[0])
            action_log_std = self.module_list_old[3].expand_as(action_mean)
            action_std = torch.exp(action_log_std)
        else:
            x = F.tanh(self.affine1(x))

            x = F.tanh(self.affine2(x))

            x = x.reshape(-1, x.shape[0])
            action_mean = self.action_mean(x)

            # action_mean = action_mean.reshape(action_mean.shape[0],-1)

            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


# modified policy network
# 1. layer normalization after first 2 activations
# 2. adding noise with a given std after layer normalizations
class PolicyLayerNorm(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PolicyLayerNorm, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        # normalization layer
        self.layer_norm = nn.LayerNorm(64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.module_list_current = [self.affine1, self.affine2, self.action_mean, self.action_log_std]

        self.module_list_old = [None] * len(
            self.module_list_current)  # self.affine1_old, self.affine2_old, self.action_mean_old, self.action_log_std_old]


    # function which produces parameter noise
    # normal distribution with mean=0,std should be given
    def parameter_noise(self, sigma):
        noise = torch.normal(mean=0, std=sigma, size=(1, 64))
        return noise

    def forward(self, x, sigma=0.0, old=False, param_noise=False):
        # print('with noise')
        if param_noise:
            if old:
                # normalization added
                x = self.layer_norm(F.tanh(self.module_list_old[0](x)))
                x = x.reshape(-1, x.shape[0])
                # parameter noise
                x += self.parameter_noise(sigma=sigma)
                # normalization added
                x = self.layer_norm(F.tanh(self.module_list_old[1](x)))
                # parameter nosie
                x += self.parameter_noise(sigma=sigma)
                action_mean = self.module_list_old[2](x)
                action_log_std = self.module_list_old[3].expand_as(action_mean)
                action_std = torch.exp(action_log_std)
            else:
                x = self.layer_norm(F.tanh(self.affine1(x)))

                x = x.reshape(-1, x.shape[0])
                x += self.parameter_noise(sigma=sigma)
                x = self.layer_norm(F.tanh(self.affine2(x)))
                x += self.parameter_noise(sigma=sigma)
                action_mean = self.action_mean(x)
                action_log_std = self.action_log_std.expand_as(action_mean)
                action_std = torch.exp(action_log_std)

            return action_mean, action_log_std, action_std
        else:
            if old:
                # normalization added
                x = F.tanh(self.module_list_old[0](x))
                x = x.reshape(-1, x.shape[0])
                # parameter noise
                # x += self.parameter_noise(sigma=sigma)
                # normalization added
                x = F.tanh(self.module_list_old[1](x))
                # parameter nosie
                # x += self.parameter_noise(sigma=sigma)
                action_mean = self.module_list_old[2](x)
                action_log_std = self.module_list_old[3].expand_as(action_mean)
                action_std = torch.exp(action_log_std)
            else:
                x = self.layer_norm(F.tanh(self.affine1(x)))
                x = x.reshape(-1, x.shape[0])
                x = self.layer_norm(F.tanh(self.affine2(x)))

                action_mean = self.action_mean(x)
                # action_mean = action_mean.reshape(1,action_mean.shape[0])

                action_log_std = self.action_log_std.expand_as(action_mean)
                action_std = torch.exp(action_log_std)

            return action_mean, action_log_std, action_std


#class for storing paths to n best policies
class sorted_list:
    def __init__(self,length):
        self.length = length
        #list which will save paths to policies
        self.paths = []
        #list which will save corresponding rewards
        self.rewards = []
        self.i=0
    def add_path(self,path,reward):
        # print(path)
        if len(self.paths)<self.length:
            self.paths.append(path)
            self.rewards.append(reward)
        else:

            #getting arguments of sorted list
            sorted_args = np.argsort(self.rewards)

            #creating new lists for sorted values
            updated_rewards = [None] * self.length
            updated_paths = [None] * self.length

            # if self.i==0:
            #     # self.rewards = np.array(self.rewards)
            #     # new_rewards = self.rewards[sorted_args]
            #     # print(new_rewards)
            #     # print(new_rewards.shape)
            #     print(self.rewards)
            #     print(sorted_args)
            #     self.i+=1

            #sorting rewards and corresponding paths in ascending order
            for i in range(self.length):

                #getting
                sorted_position = sorted_args[i]

                #updating list of sorted reward
                updated_rewards[i] = self.rewards[sorted_position]
                updated_paths[i] = self.paths[sorted_position]



            self.rewards = updated_rewards[::-1]
            self.paths = updated_paths[::-1]

            if reward > self.rewards[-1]:
                self.rewards[-1] = reward
                self.paths[-1] = path

    def get_paths(self):
        return self.paths



best_policies_paths = sorted_list(100)
best_policies_nonoise = sorted_list(100)
#searching for n best policies
# for seed in range(1,n_seeds+1):
for seed in range(11,16):
    seed_src = args.src + '/seed' + str(seed)
    data_src = seed_src + '/data'
    parameter_src = seed_src + '/parameters/'
    #getting all variations of noise
    settings = os.listdir(data_src)

    #checking all rewards for every setting
    for setting in settings:

        #loading list of reward per episode
        with open(data_src + '/' + setting,'rb') as f:
            setting_rewards = pickle.load(f)


        for episode,episode_reward in enumerate(setting_rewards):
            policy_path = parameter_src + 'episode_' + str(episode) + '/' + setting

            best_policies_paths.add_path(policy_path,episode_reward)

            if 'Nonoise' in setting:
                best_policies_nonoise.add_path(policy_path,episode_reward)






def post_evaluate(policies_dict,add_noise=False):
    '''function tests n best policies for 5 episodes each
    1. policies_dict - structure with 2 lists: path to policy parameters, reward obtained using this policy
     add_noise parameter - if true - adds parameter noise with sigma(std) same as it was used during training
    '''



    # after getting best n policies it is necessary to post evaluate all of them to choose the best one
    post_evaluation = {}

    n_test_episodes = 5
    for policy in policies_dict.paths:
        value_path = policy + "_value"
        policy_path = policy + "_policy"

        env = gym.make(args.env)
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        # loading pretrained networks
        if 'Nonoise' in policy:
            policy_layer = Policy(num_inputs,num_actions)
        else:
            policy_layer = PolicyLayerNorm(num_inputs, num_actions)
        value_layer = models.Value(num_inputs)

        policy_layer.load_state_dict(torch.load(policy_path))
        value_layer.load_state_dict(torch.load(value_path))


        reward_sum=0

        if add_noise and not 'Nonoise' in policy:

            right_part = policy.split('seed')[1]
            current_seed = right_part.split('/')[0]


            current_setting = policy.split('/')[-1]

            sigma_path = args.src + "/seed" + current_seed + "/sigma_behaviour/" + current_setting

            sigma_episode = int(policy.split('episode_')[1][0])
            with open(sigma_path,'rb') as f:
                sigmas = pickle.load(f)

            current_sigma = sigmas[sigma_episode]
            # print(current_sigma)

        for i_episode in range(n_test_episodes):

            #seed to make equivalent initial conditions for all policies
            env.seed(i_episode)
            torch.manual_seed(i_episode)

            state = env.reset()


            for t in range(1000):
                state = torch.FloatTensor(state)


                if add_noise and not 'Nonoise' in policy:
                    action_mean,_,action_std = policy_layer(state,sigma = current_sigma,param_noise=True)
                else:
                    action_mean, _, action_std = policy_layer(state)


                action = action_mean.detach().numpy()
                action = action[0,:]
                next_state,reward,done, _ = env.step(action)

                reward_sum+=reward

                if done:
                    break

        post_evaluation[policy] = reward_sum/n_test_episodes

    return post_evaluation



#best policies overall
postevaluation_noise = post_evaluate(best_policies_paths,add_noise=True)
postevaluation_noise2 = post_evaluate(best_policies_paths,add_noise=False)
#best policies without any noise
postevaluation_nonoise = post_evaluate(best_policies_nonoise,add_noise=False)



#returns path to the best policy
best_policy_noise = max(postevaluation_noise,key=lambda x:x[1])
print('Best noisy policy with ' + best_policy_noise)
print('Average reward achieved by best noisy policy(with noise during training and post evaluation): ' + str(postevaluation_noise[best_policy_noise]))

best_policy_noise2 = max(postevaluation_noise2,key=lambda x:x[1])
print('Best noisy policy with ' + best_policy_noise2)
print('Average reward achieved by best noisy policy(with noise during training, but without noise during post evaluation): ' + str(postevaluation_noise2[best_policy_noise2]))


best_policy_nonoise = max(postevaluation_nonoise,key=lambda x:x[1])
print('Policy with highest reward: ' + best_policy_nonoise)
print('Average reward achieved by policy without noise: ' + str(postevaluation_nonoise[best_policy_nonoise]))


'''Comparing average reward achieved during post evaluation with and without noise'''
post_noise = []
post_nonoise = []
policies = list(postevaluation_noise.keys())
for policy in postevaluation_noise:
    with_noise = postevaluation_noise[policy]
    without_noise = postevaluation_noise2[policy]

    post_noise.append(with_noise)
    post_nonoise.append(without_noise)

t = np.arange(len(policies))
#plotting performance
plt.plot(t,post_noise,label='post evaluation with noise')
plt.plot(t,post_nonoise,label='post evaluation without noise')
plt.legend()
plt.show()




'''here is plotting of evolution of 2 best policies: with noise during training and without'''
# env = gym.make(args.env)
# num_inputs = env.observation_space.shape[0]
# num_actions = env.action_space.shape[0]



# best_seed_noise = best_policy_noise.split('/')[2]
# best_setting_noise = best_policy_noise.split('/')[5]
#
# best_seed_nonoise = best_policy_nonoise.split('/')[2]
# best_setting_nonoise = best_policy_nonoise.split('/')[5]
#
# print(os.getcwd())
# #getting paths for best results obtained with and without results
# rewards_noise = args.src +  str(best_seed_noise) + '/data/' + best_setting_noise
# rewards_nonoise = args.src + str(best_seed_nonoise) + '/data/' + best_setting_nonoise
#
# with open(rewards_noise,'rb') as f:
#     rewards_noise_list = pickle.load(f)
#
# with open(rewards_nonoise,'rb') as f:
#     rewards_nonoise_list = pickle.load(f)
#
# t = np.arange(len(rewards_noise_list))
#
# plt.plot(t,rewards_noise_list,label=best_setting_noise)
# plt.plot(t,rewards_nonoise_list,label=best_setting_nonoise)
#
# plt.xlabel('episode')
# plt.ylabel('reward')
# plt.legend()
# plt.show()

