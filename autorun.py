import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', type=str, help='Name of the environment to run tests on')
parser.add_argument('--max-episodes', type=int, default=300, help='max number of episodes to run')
parser.add_argument('--seed', type=int,default=666,help='seed')


#this script should automate running parameter noise tests

args = parser.parse_args()

os.system("mkdir -p " + str(args.env_name) + "_tests/seed" + str(args.seed) + "/plots")
os.system("mkdir -p " + str(args.env_name) + "_tests/seed" + str(args.seed) + "/data")
# os.system("cd " + str(args.env_name) + "_tests")


# #testing with adaptive sigma
# os.system("python main.py --env-name " + str(args.env_name) + " --max-episodes " + str(args.max_episodes) + " --plot-name " + str(args.env_name)
#               + "_Adaptnoise" + 's' + str(args.seed) + "std" + str(sigma) + " --seed " + str(args.seed) + " --sigma-initial " + str(0.1) +
#               " --use-parameter-noise" + " --sigma-adaptive")

#testing 3 cases
#1. Without noise
#2. Constant sigma
#3. Linearly decreasing sigma


#testing linearly decreasing sigma

'''run without any noise'''
os.system("python main.py --env-name " + str(args.env_name)
          + " --max-episodes " + str(args.max_episodes)
          + " --plot-name " + str(args.env_name)
          + "_Nonoise" + 's' + str(args.seed)
          + " --seed " + str(args.seed))


'''run with constant sigma'''
sigmas = [0.1,0.3,0.5,0.7,1.0]

for sigma in sigmas:
    os.system("python main.py --env-name " + str(args.env_name) + " --max-episodes " + str(args.max_episodes) + " --plot-name " + str(args.env_name)
              + "_Constnoise" + 's' + str(args.seed) + "std" + str(sigma) + " --seed " + str(args.seed) + " --sigma-initial " + str(sigma) +
              " --use-parameter-noise")

'''run with linearly decreasing sigma'''
sigma0 = [0.1,0.3,0.5]

for sigma in sigma0:
    os.system("python main.py --env-name " + str(args.env_name) + " --max-episodes " + str(args.max_episodes) + " --plot-name " + str(args.env_name)
              + "_LinDecreasingNoise" + 's' + str(args.seed) + "std" + str(sigma) + " --seed " + str(args.seed) + " --sigma-initial " + str(sigma) +
              " --use-parameter-noise" + " --sigma-linear-scheduler")


os.system("cd ..")