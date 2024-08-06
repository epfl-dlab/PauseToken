from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from stable_baselines3.common.vec_env import DummyVecEnv

class LMDummyVecEnv(DummyVecEnv):
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
                self.buf_obs[key][env_idx][:len_obs] = obs
            else:
                #get the length of the observation and save it in the buffer
                len_obs = obs[key].shape[-1]
                self.buf_obs[key][env_idx][:len_obs] = obs  # type: ignore[call-overload]