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

        raw_obs, info = self.env.reset()
        obs = self._process_observations(raw_obs)
        observations.append(obs)

        for i in range(self.frame_skip - 1):
            prev_raw_obs = raw_obs
            raw_obs, reward, terminated, truncated, info = self.env.step(0) # Do nothing
            obs = self._process_observations(raw_obs, prev_raw_obs)
            observations.append(obs)

        observation = np.stack(observations)

        return observation, info

    def step(self, action):
        observations = []
        total_reward = 0
        prev_raw_obs = None
        for i in range(self.frame_skip):
            raw_obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self._process_observations(raw_obs, prev_raw_obs)
            observations.append(obs)
            total_reward += reward
            prev_raw_obs = raw_obs

        observation = np.stack(observations)

        return observation, total_reward, terminated, truncated, info

    def _process_observations(self, raw_obs, prev_raw_obs = None):
        if prev_raw_obs is not None: # if there is any previous observation
            raw_obs = np.fmax(raw_obs, prev_raw_obs) # element-wise max between the two images, over all pixel colour values
        image = Image.fromarray(raw_obs)
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
    
class NoopResetEnv(gym.Wrapper):
    """
    Gym wrapper to do a random number of no-op after reset to add randomness to the play

    :param env: Environment to wrap
    :param max_num_initial_noop_frames: The maximum number of frames to do nothing
    """
    def __init__(self, env, max_num_initial_noop_frames=30):
        super().__init__(env)
        self.max_num_initial_noop_frames = max_num_initial_noop_frames

    def reset(self):
        obs, info = self.env.reset()
        done = False
        while True:
            successful = True
            num_of_rand_initial_frames = np.random.randint(1, self.max_num_initial_noop_frames + 1)
            for i in range(num_of_rand_initial_frames):
                obs, reward, terminated, truncated, info = self.env.step(0) # no-op
                if terminated or truncated:
                    obs, info = self.env.reset()
                    successful = False
                    break
                if successful:
                    return obs, info
            

class FireResetEnv(gym.Wrapper):
    """
    Gym wrapper to manually fire the ball for the game 'Breakout'

    :param env: Environment to wrap
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(1) # Fire
        if terminated or truncated:
            return self.reset()
        return obs, info
    
class EpisodicLifeEnv(gym.Wrapper):
    """
    Gym wrapper to announce game over even if the agent losses a single life

    :param env: Environment to wrap
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, info = self.env.reset()
        self.initial_lives = self.env.unwrapped.ale.lives()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.initial_lives:
            terminated = True
        return obs, reward, terminated, truncated, info
    
class BreakoutActionTransform(gym.Wrapper):
    """
    Gym wrapper to announce game over even if the agent losses a single life

    :param env: Environment to wrap
    :
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(n=3)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        print(action)
        if action > 0:
            return action + 1
        return action