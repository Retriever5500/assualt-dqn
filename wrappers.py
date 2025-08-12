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
    def __init__(self, env, image_shape=(84, 84), stack_frames=4):
        super().__init__(env)
        self.image_shape = image_shape
        self.stack_frames = stack_frames

        obs_shape = (stack_frames, self.image_shape[0], self.image_shape[1])
        self.observation_space = gym.spaces.Box(shape=obs_shape, low=0, high=1, dtype=np.float32)

    def reset(self, *, seed = None, options = None):
        observations = []

        raw_obs, info = self.env.reset(seed=seed, options=options)
        obs = self._process_observations(raw_obs)
        observations.append(obs)

        for i in range(self.stack_frames - 1):
            raw_obs, reward, terminated, truncated, info = self.env.step(0) # Do nothing
            obs = self._process_observations(raw_obs)
            observations.append(obs)

        observation = np.stack(observations)

        return observation, info

    def step(self, action):
        observations = []
        total_reward = 0
        prev_terminated = False
        for i in range(self.stack_frames):
            raw_obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self._process_observations(raw_obs)
            observations.append(obs)
            terminated = terminated or prev_terminated
            total_reward += reward
            prev_terminated = terminated # terminated gets set True once we lose a life, so we have to apply or operation over all of the stacked frames

        observation = np.stack(observations)

        return observation, total_reward, terminated, truncated, info

    def _process_observations(self, raw_obs):
        image = Image.fromarray(raw_obs)
        image = image.convert('L')
        image = image.resize((self.image_shape[1], self.image_shape[0]))
        image_array = np.array(image).astype(np.float32)
        image_array /= 255
        return image_array

class MaxAndSkip(gym.Wrapper):
    """
    Gym wrapper to skip frames by a step of k (frame skipping get applied before AtariImage and it's frame stacking. e.g. the frames in a stack can be 0, k, 2k, etc.) and taking the max
    between the two last frames.
    
    :param env: Environment to wrap
    :param frameskip: frames to skip
    """

    def __init__(self, env, frameskip=4):
        super().__init__(env)
        assert frameskip >= 2
        self.frameskip = frameskip
        self.obs_buffer = np.zeros(shape=(2, ) + self.observation_space.shape, dtype='uint8')

    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self.frameskip):    
            obs, reward, termianted, truncated, info = self.env.step(action) # do nothing
            
            if (self.frameskip - i == 2): self.obs_buffer[0] = obs # storing the one to last frame
            elif (self.frameskip - i == 1): self.obs_buffer[1] = obs # storing the last frame
            
            total_reward += reward
            done = termianted or truncated
            
            if done: 
                break
            
        max_frame = np.fmax(self.obs_buffer[0], self.obs_buffer[1])
        
        return max_frame, total_reward, done, done, info

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
            

class FireResetWithoutEpisodicLife(gym.Wrapper):
    """
    Gym wrapper (not to be used along with EpisodicLifeEnv) to manually fire the ball for the game 'Breakout'

    :param env: Environment to wrap
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def reset(self, *, seed = None, options = None):
        while True:
            obs, info = self.env.reset(seed=seed, options=options)
            obs, reward, terminated, truncated, info = self.env.step(1) # Fire
            if terminated or truncated:
                continue
            self.lives = self.env.unwrapped.ale.lives()
            return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            obs, reward, terminated, truncated, info = self.env.step(1) # Fire when we lose a life as well
            self.lives = lives
        return obs, reward, terminated, truncated, info
    
class FireResetWithEpisodicLife(gym.Wrapper):
    """
    Gym wrapper (to be used along with EpisodicLifeEnv) to manually fire the ball for the game 'Breakout'

    :param env: Environment to wrap
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def reset(self, *, seed = None, options = None):
        while True:
            obs, info = self.env.reset(seed=seed, options=options)
            obs, reward, terminated, truncated, info = self.env.step(1) # Fire
            if terminated or truncated:
                continue
            self.lives = self.env.unwrapped.ale.lives()
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

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        if action > 0:
            return action + 1
        return action