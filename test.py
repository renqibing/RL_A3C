import torch
import torch.nn.functional as F
import time
import gym
from mycode.model import ValueNet
from mycode.model import PolicyNet, A3CMLP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_durations(rewards):
    plt.figure(2)
    plt.clf()
    plt.title("Training ...")
    plt.xlabel("episodes")
    plt.ylabel("average reward")

    plt.plot(rewards)
    plt.pause(0.001)


def test(rank, args, shared_model, counter):
    print("start testing process {}".format(args.num_processes))
    rewards = []
    torch.manual_seed(args.seed + rank)
    env = gym.make('Pendulum-v0')

    acnet = A3CMLP(env.observation_space.shape[0], env.action_space, args.stack)
    acnet.eval()

    done = True
    reward_sum = 0
    episode = 0

    while True:
        episode += 1
        # if episode > args.max_evaluation_episode:
        #     break
        if done:
            hx = torch.zeros((1, 128))
            cx = torch.zeros((1, 128))
            epc_inc = 0
            # print("updating parameters!")
        else:
            hx = hx.detach().clone()
            cx = cx.detach().clone()
        state = env.reset()
        state = torch.from_numpy(state).float()
        start_time = time.time()

        for i in range(args.num_evaluation_steps):
            env.render()
            value, mu, sigma, (hx, cx) = acnet((state, (hx, cx)))
            mu = torch.clamp(mu, -1.0, 1.0)
            action = mu.detach().numpy()[0]
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = torch.from_numpy(state).float().view(1, env.observation_space.shape[0])
            epc_inc += 1
            done = done or epc_inc > args.max_episode_length

            if i % args.num_steps == 0:
                acnet.load_state_dict(shared_model.state_dict())

        rewards.append(reward_sum.item())

        print("Time {}, train_episodes {}, FPS {:.0f}, evaluation reward {}".format(
            time.strftime("%Hh %Mm %Ss",
                          time.gmtime(time.time() - start_time)),
            counter.value, counter.value/(time.time() - start_time),
            reward_sum.item()
        ))

        plot_durations(rewards)
        time.sleep(60)
        if reward_sum > -150:
            break
        else:
            reward_sum = 0
    return
