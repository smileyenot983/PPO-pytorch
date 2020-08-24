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
import matplotlib.pyplot as plt

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
for seed in range(1,n_seeds+1):
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




#i will test for 5 episodes each policy

def post_evaluate(policies_dict):
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
            policy_layer = models.Policy(num_inputs,num_actions)
        else:
            policy_layer = models.PolicyLayerNorm(num_inputs, num_actions)
        value_layer = models.Value(num_inputs)

        policy_layer.load_state_dict(torch.load(policy_path))
        value_layer.load_state_dict(torch.load(value_path))


        reward_sum=0

        for i_episode in range(n_test_episodes):

            #seed to make equivalent initial conditions for all policies
            env.seed(i_episode)
            torch.manual_seed(i_episode)

            state = env.reset()


            for t in range(1000):
                state = torch.FloatTensor(state)
                action_mean,_,action_std = policy_layer(state)

                # action = action.data[0].numpy()
                action = action_mean.detach().numpy()
                next_state,reward,done, _ = env.step(action)

                reward_sum+=reward

                if done:
                    break

        post_evaluation[policy] = reward_sum/n_test_episodes

    return post_evaluation

#best policies overall
postevaluation_noise = post_evaluate(best_policies_paths)
#best policies without any noise
postevaluation_nonoise = post_evaluate(best_policies_nonoise)

#returns path to the best policy
best_policy_noise = max(postevaluation_noise,key=lambda x:x[1])
best_policy_nonoise = max(postevaluation_nonoise,key=lambda x:x[1])

print(best_policy_noise)
print(best_policy_nonoise)


env = gym.make(args.env)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# #plotting results obtained with best policy with noise and without noise
# policy_with_noise = models.PolicyLayerNorm(num_inputs,num_actions)
# value_with_noise = models.Value(num_inputs)
# policy_with_noise.load_state_dict(torch.load(best_policy_noise + "_policy"))
# value_with_noise.load_state_dict(torch.load(best_policy_noise + "_value"))
#
#
# policy_without_noise = models.Policy(num_inputs,num_actions)
# value_without_noise = models.Value(num_inputs)
# policy_without_noise.load_state_dict(torch.load(best_policy_nonoise + "_policy"))
# value_without_noise.load_state_dict(torch.load(best_policy_nonoise + "_value"))
#
# #now evaluating them during 300 episodes each to plot obtained rewards
# n_final_episodes = 300
# def test_policy(policy_net):
#     policy_rewards = []
#     for i_episode in range(n_final_episodes):
#         episode_reward=0
#         env = gym.make(args.env)
#         env.seed(i_episode)
#         torch.manual_seed(i_episode)
#
#         state =env.reset()
#
#         for t in range(1000):
#             state = torch.FloatTensor(state)
#             action_mean, _, action_std = policy_net(state)
#
#
#             action = action_mean.detach().numpy()
#             next_state, reward, done, _ = env.step(action)
#
#             episode_reward += reward
#
#             if done:
#                 break
#
#         policy_rewards.append(episode_reward)
#
#     return policy_rewards


# results_with_noise = test_policy(policy_with_noise)
# results_without_noise = test_policy(policy_without_noise)
#
# t = np.arange(n_final_episodes)
# plt.scatter(t,results_with_noise,label='with noise')
# plt.scatter(t,results_without_noise,label='without noise')
# plt.legend()
# plt.show()



best_seed_noise = best_policy_noise.split('/')[2]
best_setting_noise = best_policy_noise.split('/')[5]

best_seed_nonoise = best_policy_nonoise.split('/')[2]
best_setting_nonoise = best_policy_nonoise.split('/')[5]

print(os.getcwd())
#getting paths for best results obtained with and without results
rewards_noise = args.src +  str(best_seed_noise) + '/data/' + best_setting_noise
rewards_nonoise = args.src + str(best_seed_nonoise) + '/data/' + best_setting_nonoise

with open(rewards_noise,'rb') as f:
    rewards_noise_list = pickle.load(f)

with open(rewards_nonoise,'rb') as f:
    rewards_nonoise_list = pickle.load(f)

t = np.arange(len(rewards_noise_list))

plt.plot(t,rewards_noise_list,label=best_setting_noise)
plt.plot(t,rewards_nonoise_list,label=best_setting_nonoise)

plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()

