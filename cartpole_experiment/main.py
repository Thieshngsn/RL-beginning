import random
from abc import ABC
from typing import TypeVar
import random
import gym

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == '__main__':
    env = RandomActionWrapper(gym.make('CartPole-v1'))
    reward = 0.0
    steps = 0.0
    obs = env.reset()

    while True:
        env.render()
        action = env.action_space.sample()
        obs, _reward, done, _ = env.step(action)
        print(obs)
        reward += _reward
        steps += 1
        if done:
            break

    print(f'reward: {reward}, steps: {steps}')