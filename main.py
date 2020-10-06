import argparse
import sys
import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import scipy.optimize
from gym import wrappers

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from models import Policy, Value, ActorCritic, PolicyLayerNorm
from replay_memory import Memory
from running_state import ZFilter

import matplotlib.pyplot as plt
# from utils import *
import pickle
import cartpole_swingup

import pybulletgym

# import sparseMuJoCo



torch.set_default_tensor_type('torch.DoubleTensor')
PI = torch.DoubleTensor([3.1415926])

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
# parser.add_argument('--l2_reg', type=float, default=1e-3, metavar='G',
#                     help='l2 regularization regression (default: 1e-3)')
# parser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',
#                     help='max kl value (default: 1e-2)')
# parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
#                     help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='batch size (default: 5000)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--entropy-coeff', type=float, default=0.0, metavar='N',
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='Clipping for PPO grad')
parser.add_argument('--use-joint-pol-val', action='store_true',
                    help='whether to use combined policy and value nets')
parser.add_argument('--use-parameter-noise', action='store_true',
                    help='add noise to weights of actor network')
parser.add_argument('--max-episodes', type=int,default=2000,help='max number of episodes to train ')
# parser.add_argument('--layer-normalization', type=bool, default=False,
#                     help='layer normalization for first 2 layers in order to use parameter noise')
parser.add_argument('--plot-name', type=str, default='PPO rewards',
                    help='name of plot')
parser.add_argument('--sigma-adaptive',action='store_true',help='whether to use adaptive sigma or not')
parser.add_argument('--sigma-initial',default=0.1,help='initial value for noise')
parser.add_argument('--sigma-linear-scheduler', action='store_true')
parser.add_argument('--sigma-exponential-scheduler', action='store_true')

parser.add_argument('--post-evaluation-interval', type=int, default=5, metavar='N',
                    help='interval between postevaluations')

parser.add_argument('--parameter-save-interval', type=int, default=10, metavar='N',
                    help='interval between saving parameters of networks')


# parser.add_argument('--plot-folder', type=str, default='./plots', help='folder for storing plots')


args = parser.parse_args()
print(args.use_parameter_noise)
env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

if args.use_joint_pol_val:
    ac_net = ActorCritic(num_inputs, num_actions)
    opt_ac = optim.Adam(ac_net.parameters(), lr=0.001)
#Here if we are using parameter noise we should use modified policy network
elif args.use_parameter_noise:
    policy_net = PolicyLayerNorm(num_inputs, num_actions)
    value_net = Value(num_inputs)
    opt_policy = optim.Adam(policy_net.parameters(), lr=0.001)
    opt_value = optim.Adam(value_net.parameters(), lr=0.001)
else:
    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)
    opt_policy = optim.Adam(policy_net.parameters(), lr=0.001)
    opt_value = optim.Adam(value_net.parameters(), lr=0.001)

# def select_action(state,sigma):
#     state = torch.from_numpy(state).unsqueeze(0)
#     if args.use_parameter_noise:
#         action_mean, _, action_std = policy_net(Variable(state),sigma,param_noise=True)
#     else:
#         action_mean, _, action_std = policy_net(Variable(state))
#     action = torch.normal(action_mean, action_std)
#     return action

def select_action(policy,state,sigma=1e-5,add_noise=False):
    state = torch.from_numpy(state).unsqueeze(0)
    state = state.double()
    if args.use_parameter_noise and add_noise:
        action_mean, _, action_std = policy(Variable(state),sigma)
    else:
        action_mean, _, action_std = policy(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


def select_action_actor_critic(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std, v = ac_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)

def update_params_actor_critic(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    action_means, action_log_stds, action_stds, values = ac_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # kloldnew = policy_net.kl_old_new() # oldpi.pd.kl(pi.pd)
    # ent = policy_net.entropy() #pi.pd.entropy()
    # meankl = torch.reduce_mean(kloldnew)
    # meanent = torch.reduce_mean(ent)
    # pol_entpen = (-args.entropy_coeff) * meanent

    action_var = Variable(actions)
    # compute probs from actions above
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old, values_old = ac_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    ac_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_ac.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()

    vf_loss1 = (values - targets).pow(2.)
    vpredclipped = values_old + torch.clamp(values - values_old, -args.clip_epsilon, args.clip_epsilon)
    vf_loss2 = (vpredclipped - targets).pow(2.)
    vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

    total_loss = policy_surr + vf_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm(ac_net.parameters(), 40)
    opt_ac.step()


def update_params(batch,sigma):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    opt_value.zero_grad()
    value_loss = (values - targets).pow(2.).mean()
    value_loss.backward()
    opt_value.step()

    # kloldnew = policy_net.kl_old_new() # oldpi.pd.kl(pi.pd)
    # ent = policy_net.entropy() #pi.pd.entropy()
    # meankl = torch.reduce_mean(kloldnew)
    # meanent = torch.reduce_mean(ent)
    # pol_entpen = (-args.entropy_coeff) * meanent

    action_var = Variable(actions)

    if args.use_parameter_noise:

        action_means, action_log_stds, action_stds = policy_net(Variable(states),sigma,param_noise=True)
    else:
        action_means, action_log_stds, action_stds = policy_net(Variable(states))
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)


    if args.use_parameter_noise:
        action_means_old, action_log_stds_old, action_stds_old = policy_net(Variable(states),sigma, old=True,param_noise=True)

    else:
        action_means_old, action_log_stds_old, action_stds_old = policy_net(Variable(states), old=True)

    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    policy_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_policy.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    opt_policy.step()


def post_evaluate(models_path, sigma, n_post_episodes=5,add_noise=False):
    # print('----------------Post evaluation----------------')

    policy_path = models_path + "_policy"
    value_path = models_path + "_value"

    if args.use_parameter_noise:
        policy_post = PolicyLayerNorm(num_inputs, num_actions)
        value_post = Value(num_inputs)

    else:
        policy_post = Policy(num_inputs, num_actions)
        value_post = Value(num_inputs)

    # print('------------------')
    value_post.load_state_dict(torch.load(value_path))
    policy_post.load_state_dict(torch.load(policy_path))


    reward_post = 0

    for i in range(n_post_episodes):
        state = env.reset()
        # state = running_state(state)
        for t in range(1000):

            if args.use_parameter_noise and add_noise:
                action = select_action(policy_post, state, sigma,add_noise=True)

            else:
                action = select_action(policy_post, state)
            action = action.data[0].numpy()

            next_state, reward, done, _ = env.step(action)

            reward_post += reward

            # next_state = running_state(next_state)

            if done:
                break

            # state = running_state(next_state)
            state = next_state

    print('___Post evaluation reward___')
    print(reward_post / n_post_episodes)

    return reward_post / n_post_episodes

#calculating distance between 2 policies
def policies_distance(batch,sigma):

    states = torch.Tensor(batch.state)

    #forward propagation without parameter noise
    action_means, action_log_stds, action_stds = policy_net(Variable(states),sigma,param_noise=False)
    #forward propagation with parameter noise
    action_means_perturbed, _,_ = policy_net(Variable(states),sigma,param_noise=True)

    #euclidean distance between means of perturbed and non perturbed policies
    distance = torch.sum((action_means - action_means_perturbed)**2)**0.5
    return distance


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []

#HYPERPARAMETERS FOR PARAMETER NOISE
#initial noise std
sigma = float(args.sigma_initial)
#scalefactor for adapting noise, we will multiply or divide by this number to adapt sigma for noise
sigma_scalefactor = 1.01
#distance threshold for adapting noise
distance_threshold = 1
# distance_threshold = 0.001 * args.batch_size
#perturbation frequency
perturbation_timestep = 2

#linearly decreasing sigma
sigma_schedulefactor = 0.999

sigma_behaviour = []

rewards_returned = []

#TODO:
# 1. add post evaluation after every n_th episode and save obtained reward in binary

post_evaluation_noise1 = [] #---with noise during post evaluation
post_evaluation_noise2 = [] #---without noise during post evaluation
post_evaluation_nonoise = [] #---without noise during training

for i_episode in range(args.max_episodes):
# for i_episode in count(1):
    #create memory to save experience
    memory = Memory()
    num_steps = 0
    reward_batch = 0
    num_episodes = 0

    if args.use_parameter_noise:
        # saving behaviour of sigma
        sigma_behaviour.append(sigma)

    #default batch size = 5000
    while num_steps < args.batch_size:

        state = env.reset()
        # state = running_state(state)

        reward_sum = 0
        for t in range(1000): # Don't infinite loop while learning
            if args.use_joint_pol_val:
                action = select_action_actor_critic(state)
            else:
                # action = select_action(state,sigma)
                action = select_action(policy_net,state,sigma,add_noise=True)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            # next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render and i_episode >= 20:
                env.render()
            if done:
                break

            state = next_state

        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

        # print('came')

    #avergae policy reward
    reward_batch /= num_episodes
    batch = memory.sample()

    models_path = str(args.env_name) + "_tests" + "/seed" + str(args.seed) + "/parameters/episode_" + str(
        i_episode) + "/" + str(args.plot_name)
    torch.save(policy_net.state_dict(), models_path + "_policy")
    torch.save(value_net.state_dict(), models_path + "_value")



    if i_episode % args.parameter_save_interval==0:

        if args.use_parameter_noise:

            post_evaluation_noise1.append(post_evaluate(models_path, sigma, add_noise=True))
            post_evaluation_noise2.append(post_evaluate(models_path, sigma, add_noise=False))

        else:
            post_evaluation_nonoise.append(post_evaluate(models_path, sigma, add_noise=False))

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {} \t Average reward {:.2f} \t Sigma {}'.format(
            i_episode, reward_sum, reward_batch, sigma))




    if args.use_joint_pol_val:
        update_params_actor_critic(batch)
    else:
        update_params(batch,sigma)
    rewards_returned.append(reward_batch)

    #adaptive std for parameter noise
    #1. computes current distance between perturbed and non perturbed policies
    #2. if distance higher than threshold -> reduce std, if less -> increase
    if args.use_parameter_noise and args.sigma_adaptive:
        current_distance = policies_distance(memory.sample(), sigma)
        # print(current_distance)

        if current_distance > distance_threshold:
            sigma /= sigma_scalefactor
        else:
            sigma *= sigma_scalefactor

    if args.use_parameter_noise and args.sigma_linear_scheduler:
        sigma *= sigma_schedulefactor

    if args.use_parameter_noise and args.sigma_exponential_scheduler:
        sigma = sigma ** sigma_schedulefactor


t = np.arange(len(rewards_returned))

if args.use_parameter_noise:
    #saving data into a binary file
    with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/sigma_behaviour/' + str(args.plot_name), 'wb') as f:
        pickle.dump(sigma_behaviour,f)

    # with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/post_evaluation/' + 'noisy_policy_noise_post', 'wb') as f:
    #     pickle.dump(post_evaluation_noise1,f)
    #
    # with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/post_evaluation/' + 'noisy_policy_nonoise_post', 'wb') as f:
    #     pickle.dump(post_evaluation_noise2,f)


    # print('______________________________________')
    # setting_name = args.plot_name.split("s" + str(args.seed))
    # print(setting_name)
    # print('______________________________________')
    with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/post_evaluation/' + str(args.plot_name) + '_noise_post', 'wb') as f:
        pickle.dump(post_evaluation_noise1,f)

    with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/post_evaluation/' + str(args.plot_name) +'_nonoise_post', 'wb') as f:
        pickle.dump(post_evaluation_noise2,f)

else:
    with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/post_evaluation/' + str(args.plot_name), 'wb') as f:
        pickle.dump(post_evaluation_nonoise,f)




#saving data into a binary file
with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/data/' + str(args.plot_name), 'wb') as f:
    pickle.dump(rewards_returned,f)

#loading data from binary file
with open(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/data/' + str(args.plot_name), 'rb') as f:
    rewards_returned = pickle.load(f)

plt.scatter(t,rewards_returned)
plt.ylabel('Rewards')
plt.savefig(str(args.env_name) + "_tests" + '/seed' + str(args.seed) + '/plots/' + str(args.plot_name) + '.png')




