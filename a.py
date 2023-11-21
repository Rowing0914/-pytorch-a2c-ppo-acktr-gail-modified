import os
import gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

vec_env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
print(vec_env.observation_space)
asdf
import pudb; pudb.start()
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)