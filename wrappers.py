import numpy as np
import gymnasium as gym
from PIL import Image
from ale_py import ALEInterface


class AtariImage(gym.Wrapper):
    """
    Gym wrapper to preprocess the environments observations (frames)
    The wrapper applies frameskip and stacks frames together
    The same action is taken in each frame of a stack

    :param env: Environment to wrap
    :param image_shape: The output shape of the image
    :param frame_skip: The amount of frames that stack, also the same action is applied
    """
    def __init__(self, env, image_shape=(84, 84), frame_skip=4):
        super().__init__(env)
        self.image_shape = image_shape
        self.frame_skip = frame_skip

        obs_shape = (frame_skip, self.image_shape[0], self.image_shape[1])
        self.observation_space = gym.spaces.Box(shape=obs_shape, low=0, high=1, dtype=np.float32)

    def reset(self):
        observations = []

        obs, info = self.env.reset()
        obs = self._process_observations(obs)
        observations.append(obs)

        for i in range(self.frame_skip - 1):
            obs, reward, terminated, truncated, info = self.env.step(0) # Do nothing
            obs = self._process_observations(obs)
            observations.append(obs)

        observation = np.stack(observations)

        return observation, info

    def step(self, action):
        observations = []
        total_reward = 0
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self._process_observations(obs)
            observations.append(obs)
            total_reward += reward

        observation = np.stack(observations)

        return observation, total_reward, terminated, truncated, info

    def _process_observations(self, obs):
        image = Image.fromarray(obs)
        image = image.convert('L')
        image = image.resize((self.image_shape[1], self.image_shape[0]))
        image_array = np.array(image).astype(np.float32)
        image_array /= 255
        return image_array


class ClipReward(gym.Wrapper):
    """
    Gym wrapper to clip rewards

    :param env: Environment to wrap
    :param min_reward: The minimum reward
    :param max_reward: The maximum reward
    """
    def __init__(self, env, min_reward=-1, max_reward=1):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(np.clip(reward, self.min_reward, self.max_reward))

        return obs, reward, terminated, truncated, info
