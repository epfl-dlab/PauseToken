from typing import Callable, List
from gymnasium.core import Env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

class LMDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], Env]], filler_token: int = -100):
        super().__init__(env_fns)
        self.filler_token = filler_token
        
        #iterate buf_obs (an ordered dict) and fille array with filler_token
        for key in self.buf_obs:
            self.buf_obs[key].fill(self.filler_token)
        
    def set_filler_token(self, filler_token: int):
        previous_filler_token = self.filler_token
        self.filler_token = filler_token
        
        for key in self.buf_obs:
            self.buf_obs[key][self.buf_obs[key]==previous_filler_token] = self.filler_token
        
    """ Vectorized environment for language model environments. This class is a subclass of DummyVecEnv and is used to handle observations of variable length. It is used to handle environments where the observation space is a sequence of tokens of variable length."""
    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        """ Save the observation in the buffer
        
        :param env_idx: Environment index
        :type env_idx: int
        :param obs: Observation
        :type obs: VecEnvObs
        """
        for key in self.keys:    
            if key is None:
                #get the length of the observation and save it in the buffer
                len_obs = obs.shape[-1]
                
            else:
                #get the length of the observation and save it in the buffer
                len_obs = obs[key].shape[-1]

            self.buf_obs[key][env_idx][:len_obs] = obs
            self.buf_obs[key][env_idx][len_obs:] = self.filler_token
            
    def set_stage(self, stage: str, **kwargs):
        """ Set the stage of the environment
        
        :param stage: Stage of the environment
        :type stage: str
        """
        for env in self.envs:
            env.set_stage(stage, **kwargs)        

    
