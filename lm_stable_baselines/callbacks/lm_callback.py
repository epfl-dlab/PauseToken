from stable_baselines3.common.callbacks import BaseCallback  
from lm_stable_baselines.utils import add_filler_tokens
import numpy as np
from tqdm import tqdm
class LMCallback(BaseCallback):
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        
        arrays_to_reshape = ["actions", "buffer_actions", "new_obs"]
        max_len = len(self.locals["self"].action_space)
        filler_token = self.locals["self"].policy.filler_token
        for key in arrays_to_reshape:
            if key in self.locals:
                self.locals[key] = add_filler_tokens(
                    array=self.locals[key],
                    max_tokens=max_len,
                    filler_token=filler_token,
                )
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                infos[i]["terminal_observation"] = \
                    add_filler_tokens(
                        infos[i]["terminal_observation"],
                        max_len,
                        filler_token,
                    )
        return True
    
    
class StarProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    pbar: tqdm

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )
        
    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        num_collected_episodes = 0
        rew_thrsh = self.locals["replay_buffer"].advantage_threshold
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                if rew_thrsh is not None:
                    above_thrsh = self.locals["rewards"][idx] > rew_thrsh
                    num_collected_episodes += int(above_thrsh)
                else:
                    # Update stats
                    num_collected_episodes += 1

        if num_collected_episodes > 0:
            self.pbar.update(num_collected_episodes)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()


class EnvironmentPortionBaseUpdate():
    """
    Update the portion of the environment actions that is used for training, linearly according to the current timestep.
    """
    
    def __init__(self,):
        self.locals = {}
        self.globals = {}

    def on_outer_loop_start(self):
        raise NotImplementedError

    def on_training_start(self, locals_, globals_) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_

    def update_locals(self, locals_) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)


class EnvironmentPortionBetaUpdate(EnvironmentPortionBaseUpdate):
    """
    Update the portion of the environment actions that is used for training, linearly according to the current timestep.
    """
    
    def __init__(self, init_alpha, init_beta, final_alpha, final_beta, warmup_timesteps, total_timesteps):
        super(EnvironmentPortionBetaUpdate, self).__init__()
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.final_alpha = final_alpha
        self.final_beta = final_beta
        self.warmup_timesteps = warmup_timesteps
        self.total_timesteps = total_timesteps
        self.alpha = self.init_alpha
        self.beta = self.init_beta

    def update(self,):
        if self.current_step < self.warmup_timesteps:
            self.alpha = self.init_alpha
            self.beta = self.init_beta
        elif self.current_step < self.total_timesteps:
            self.alpha = self.init_alpha + \
                (self.final_alpha - self.init_alpha) * (self.current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps)
            self.beta = self.init_beta + \
                (self.final_beta - self.init_beta) * (self.current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps)
        else:
            self.alpha = self.final_alpha
            self.beta = self.final_beta

    def _sample_portion(self, size=1):
        sample = np.random.beta(self.alpha, self.beta, size)
        return sample
    
    def on_outer_loop_start(self):
        environments = self.locals['self'].rl_algorithm.env.envs
        self.current_step = self.locals['self'].current_outer_loop
        self.total_timesteps = self.locals['self'].n_outer_loops
        self.update()
        portion = self._sample_portion(len(environments))
        for env, p in zip(environments, portion):
            env.set_portion(p)


class EnvironmentPortionLinearUpdate(EnvironmentPortionBaseUpdate):
    """
    Update the portion of the environment actions that is used for training, linearly according to the current timestep.
    """
    
    def __init__(self, init_portion, final_portion, warmup_timesteps, total_timesteps):
        super(EnvironmentPortionLinearUpdate, self).__init__()
        self.init_portion = init_portion
        self.final_portion = final_portion
        self.warmup_timesteps = warmup_timesteps
        self.total_timesteps = total_timesteps

    def update(self,):
        if self.current_step < self.warmup_timesteps:
            self.portion = self.init_portion
        elif self.current_step < self.total_timesteps:
            self.portion = self.init_portion + \
                (self.final_portion - self.init_portion) * (self.current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps)
        else:
            self.portion = self.final_portion
    
    def on_outer_loop_start(self):
        environments = self.locals['self'].rl_algorithm.env.envs
        self.current_step = self.locals['self'].current_outer_loop
        self.total_timesteps = self.locals['self'].n_outer_loops
        self.update()
        for env in environments:
            env.set_portion(self.portion)