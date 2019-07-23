from mycode.model import ValueNet
from mycode.model import PolicyNet, A3CMLP
from mycode.utils import normal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import math
import numpy as np


def ensure_share_grad(model,shared_model):
    for param,shared_param in zip(model.parameters(),
                               shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer = None):
    print("Start training process {}".format(rank))
    torch.manual_seed(args.seed + rank)

    env = gym.make('Pendulum-v0')
    env.seed(args.seed + rank)

    acnet = A3CMLP(env.observation_space.shape[0], env.action_space, args.stack)

    # episode = 1
    if optimizer is None:
        acoptimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    state = env.reset()
    state = torch.from_numpy(state).float()
    done = True

    shared_model.train()

    # total_loss = []
    # reward_sum = 0

    for episode in range(args.max_episode):

        # load latest parameters
        acnet.load_state_dict(shared_model.state_dict())

        rewards = []  # get from env
        values = []  # get from value net
        entropies = []  # get entropy of sigma
        log_probs = []  # get log probability of policy

        # start to generate experience
        if args.render:
            env.render(mode='human')

        if done:
            hx = torch.zeros(1, 128)
            cx = torch.zeros(1, 128)
        else:
            hx = hx.detach().clone()
            cx = cx.detach().clone()

        for t in range(args.num_steps):

            value, mu, sigma, (hx, cx) = acnet((state, (hx, cx)))

            mu = torch.clamp(mu, -1.0, 1.0)
            sigma = F.softplus(sigma) + 1e-5
            eps = torch.randn(mu.size())
            pi = np.array([math.pi])
            pi = torch.from_numpy(pi).float()
            action = mu + eps * sigma.sqrt()
            prob = normal(action.detach().clone(), mu, sigma)  # optimize mu and sigma to train the function more deterministically

            entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
            entropies.append(entropy)

            log_prob = (prob + 1e-6).log()
            log_probs.append(log_prob)

            state, reward, done, _ = env.step(action.detach().numpy()[0])
            state = torch.from_numpy(state).float()
            # reward = max(min(reward, 1.0), -1.0)
            args.epc_inc += 1
            done = done or args.epc_inc >= args.max_episode_length

            rewards.append(reward)
            values.append(value)


            if done:
                break

        with lock:
            counter.value += 1
        # from the last state, calculate value and accumulate gradients
        if done:
            state = env.reset()
            state = torch.from_numpy(state).float()
            args.epc_inc = 0
        R = torch.zeros(1, 1)

        if not done:
            value, _, _, _ = acnet((state, (hx, cx)))
            R = value.detach().clone()

        values.append(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = rewards[i] + args.gamma * R
            advantage = R - values[i]  # baseline as v(s) current value as MC evaluation
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # generalized advantage estimation policy-loss
            delta_t = rewards[i] + args.gamma * values[i + 1].detach() - values[i].detach()
            # there to optimize parameters of policy network,
            # estimation of advantage function is detached from the value net
            gae = args.gae_lambda * args.gamma * gae + delta_t

            policy_loss = policy_loss - \
                          args.entropy_coef * entropies[i].sum() - \
                          log_probs[i].sum() * gae
        total_loss = value_loss + 0.5 * policy_loss

        # total_loss.append((value_loss + 0.5 * policy_loss).item())
        # print(rewards)

        if episode % 50 == 0:
            print("thread {} training epoch {} sum of reward {}".format(rank, episode, sum(rewards)))
            print("thread {} training epoch {} total loss {}".format(rank, episode, total_loss.item()))
        acnet.zero_grad()  # clear gradients of network per thread
        total_loss.backward()
        ensure_share_grad(acnet, shared_model)
        acoptimizer.step()  # optimizer for shared model






