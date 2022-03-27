from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
from env import Pong_env
import time
import joblib


dir_path = os.path.dirname(os.path.realpath(__file__))
HIDDEN_SIZE = 20
BATCH_SIZE = 128
PERCENTILE = 85

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
        
        
        
class Brain(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        print(obs_size, hidden_size, n_actions)
        super(Brain, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    
    
    
def show_single(env, net, fps=100):
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    _time = 1/fps
    cum_reward = 0
    while True:
        start = time.time()
        env.show()
        obs_v = torch.FloatTensor([obs])
        #print(obs_v)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done = env.step(action)
        cum_reward += reward
        if is_done:
            print('testing reward: ', cum_reward)
            return
        obs = next_obs
        end = time.time()
        took = start-end
        if took < _time:
            time.sleep(_time-took)
            
        
        
def fps(fun, fps, *args):
    start = time.time()
    fun(*args)
    _time = 1/fps
    end = time.time()
    took = start-end
    if took < _time:
        time.sleep(_time-took)
    
    
        
        
def iterate_batches(env, net, batch_size, show=False):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        if show:
            env.show()
            time.sleep(0.02)
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:

            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean
        
        
if __name__ == '__main__':
    print(torch.cuda.is_available())
    env = Pong_env()
    net = Brain(env.observation_space_sample.shape[0], 5, env.action_space_sample.shape[0])
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-Pong")
    batches = iterate_batches(env, net, BATCH_SIZE)
    for iter_no, batch in enumerate(batches):
        obs_v, acts_v, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE)
        if iter_no % 50 == 0:
            for i in range(5):
                show_single(env, net, 200)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 5000:
            print("Solved!")
            break
    show_single(env, net)
    obs_v = env.reset()
    is_done = False
    sm = nn.Softmax(dim=1)
    cum_reward = 0
    tx = []
    bx = []
    while not is_done:
        obs_v = torch.FloatTensor([obs_v])
        tx.append(obs_v.data.numpy()[0][0])
        bx.append(obs_v.data.numpy()[0][1])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        #print(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)
        obs_v, reward, is_done = env.step(action)
        cum_reward += reward
    print(cum_reward)
    plt.plot(tx[-3000:], label='Distance Tile Ball')
    plt.plot(bx[-3000:], label='Ball Y')
    plt.legend()
    plt.show()
    writer.close()
    joblib.dump(net, 'pong/pong.joblib')
    torch.save(net.state_dict(), 'pong\\pong.pth')
    
    