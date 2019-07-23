import argparse

import gym
import torch
import time
from mycode.model import ValueNet
from mycode.model import PolicyNet, A3CMLP
from mycode.train import train
from mycode.test import test

import torch.multiprocessing as mp

import matplotlib.pyplot as plt

exitFlag = 0

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--gae-lambda', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--env-name', type=str, default='Pendulum-v0')
parser.add_argument('--value-loss-coef', type=float, default=0.5)
parser.add_argument('--policy-loss-coef', type=float, default=1.00)
parser.add_argument('--entropy-coef', type=float, default=1e-4)
# parser.add_argument('--num-steps', type=int, default=10)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--max-episode', type=int, default=1000)
parser.add_argument('--max-episode-length', type=int, default=200)
parser.add_argument('--max-evaluation-episode', type=int, default=100)
parser.add_argument('--num-evaluation-steps', type=int, default=200)
parser.add_argument('--num-processes', type=int, default=4)
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--stack', type=int, default=1, help='number of observations stacked')
parser.add_argument('--epc_inc', type=int, default=0)



if __name__ == '__main__':

    args = parser.parse_args()

    args.num_processes = 4

    plt.ion()

    torch.manual_seed(args.seed)
    env = gym.make(args.env_name)

    shared_model = A3CMLP(env.observation_space.shape[0], env.action_space, args.stack)
    shared_model.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)
    time.sleep(0.1)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        p.join()
        time.sleep(0.1)

    env.reset()
    env.close()
    plt.ioff()
    plt.show()


