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

    def reset(self, *, seed = None, options = None):
        observations = []

        raw_obs, info = self.env.reset(seed=seed, options=options)
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

    def reset(self, *, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        while True:
            successful = True
            num_of_rand_initial_frames = np.random.randint(1, self.max_num_initial_noop_frames + 1)
            for i in range(num_of_rand_initial_frames):
                obs, reward, terminated, truncated, info = self.env.step(0) # no-op
                if terminated or truncated:
                    obs, info = self.env.reset(seed=seed, options=options)
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

    def reset(self, *, seed = None, options = None):
        successful_reset = False
        while not successful_reset:
            obs, info = self.env.reset(seed=seed, options=options)
            obs, reward, terminated, truncated, info = self.env.step(1) # Fire
            if terminated or truncated:
                continue
            else:
                successful_reset = True
        return obs, info
    
class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    # adapted from github.com/iewug/Atari-DQN
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True

        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)

        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info
    
class BreakoutActionTransform(gym.Wrapper):
    """
    Gym wrapper to map the actions as follows {0->0(Noop), 1->2(Right), 2->3(Left)}

    :param env: Environment to wrap
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(n=3)

    def action(self, action):
        if action > 0:
            return action + 1
        return action