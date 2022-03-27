import os
from env import Pong_env
import time
import joblib
from agent import Brain, show_single
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
env = Pong_env()
net = joblib.load('pong/pong.joblib')

for i in range(5):
    show_single(env, net, 200)