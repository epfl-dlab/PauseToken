from stable_baselines3.common.callbacks import BaseCallback  
from lm_stable_baselines.utils import add_filler_tokens
import numpy as np
from tqdm import tqdm

from functools import partial

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


class EnvironmentPortionBaseUpdate(BaseCallback):
    """
    Update the portion of the environment actions that is used for training, linearly according to the current timestep.
    """
    
    def __init__(self) -> None:
        super().__init__()

    def _on_training_start(self) -> None:
        self.update()

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        self.update()
        return True

    def update(self,):
        raise NotImplementedError



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
        self.n_outer_loops_to_warmup = warmup_timesteps
        self.n_outer_loops_to_anneal = total_timesteps
    
    def update(self,):
        self.current_outer_loop = self.locals['self'].current_outer_loop
        # self.n_outerloop = self.locals['self'].n_outer_loops
        self.total_steps_in_each_outerloop = self.locals['self']._total_timesteps
        self.current_step = self.locals['self'].num_timesteps # this goes from 0 to total_timesteps
        
        current_total_step = self.current_outer_loop * self.total_steps_in_each_outerloop + self.current_step
        total_total_step = self.n_outer_loops_to_anneal * self.total_steps_in_each_outerloop
        total_warmup_step = self.n_outer_loops_to_warmup * self.total_steps_in_each_outerloop

        if current_total_step < total_warmup_step:
            self.alpha = self.init_alpha
            self.beta = self.init_beta
        elif current_total_step < total_total_step:
            self.alpha = self.init_alpha + \
                (self.final_alpha - self.init_alpha) * (current_total_step - total_warmup_step) / (total_total_step - total_warmup_step)
            self.beta = self.init_beta + \
                (self.final_beta - self.init_beta) * (current_total_step - total_warmup_step) / (total_total_step - total_warmup_step)
        else:
            self.alpha = self.final_alpha
            self.beta = self.final_beta
        self.locals['self'].logger.record(f"{self.locals['self'].env.envs[0].stage}/alpha", self.alpha)
        self.locals['self'].logger.record(f"{self.locals['self'].env.envs[0].stage}/beta", self.beta)
        
        # use partials to not define the function every time
        self.portion_dist = partial(np.random.default_rng().beta, a=self.alpha, b=self.beta,)
        environments = self.locals['self'].env.envs
        for env in environments:
            env.set_portion(self.portion_dist)
        
        


class EnvironmentPortionLinearUpdate(EnvironmentPortionBaseUpdate):
    """
    Update the portion of the environment actions that is used for training, linearly according to the current timestep.
    """
    
    def __init__(self, lower_bound_init_portion, lower_bound_final_portion, upper_bound_init_portion, upper_bound_final_portion, warmup_timesteps, total_timesteps):
        super(EnvironmentPortionLinearUpdate, self).__init__()
        self.lower_bound_init_portion = lower_bound_init_portion
        self.lower_bound_final_portion = lower_bound_final_portion
        self.upper_bound_init_portion = upper_bound_init_portion
        self.upper_bound_final_portion = upper_bound_final_portion
        self.n_outer_loops_to_warmup = warmup_timesteps
        self.n_outer_loops_to_anneal = total_timesteps

        self.lower_bound = self.lower_bound_init_portion
        self.upper_bound = self.upper_bound_init_portion

        assert self.lower_bound_init_portion <= self.upper_bound_init_portion, "Initial portion must be less than or equal to final portion"
        assert self.lower_bound_final_portion <= self.upper_bound_final_portion, "Initial portion must be less than or equal to final portion"

    def update(self,):
        self.current_outer_loop = self.locals['self'].current_outer_loop
        # self.n_outerloop = self.locals['self'].n_outer_loops
        self.total_steps_in_each_outerloop = self.locals['self']._total_timesteps
        self.current_step = self.locals['self'].num_timesteps # this goes from 0 to total_timesteps
        
        current_total_step = self.current_outer_loop * self.total_steps_in_each_outerloop + self.current_step
        total_total_step = self.n_outer_loops_to_anneal * self.total_steps_in_each_outerloop
        total_warmup_step = self.n_outer_loops_to_warmup * self.total_steps_in_each_outerloop

        if current_total_step < total_warmup_step:
            self.lower_bound = self.lower_bound_init_portion
            self.upper_bound = self.upper_bound_init_portion
        elif current_total_step < total_total_step:
            self.lower_bound = self.lower_bound_init_portion + \
                (self.lower_bound_final_portion - self.lower_bound_init_portion) * (current_total_step - total_warmup_step) / (total_total_step - total_warmup_step)
            self.upper_bound = self.upper_bound_init_portion + \
                (self.upper_bound_final_portion - self.upper_bound_init_portion) * (current_total_step - total_warmup_step) / (total_total_step - total_warmup_step)
        else:
            self.lower_bound = self.lower_bound_final_portion
            self.upper_bound = self.upper_bound_final_portion

        self.locals['self'].logger.record(f"{self.locals['self'].env.envs[0].stage}/lower_bound", self.lower_bound)
        self.locals['self'].logger.record(f"{self.locals['self'].env.envs[0].stage}/upper_bound", self.upper_bound)

        self.portion_dist = partial(np.random.default_rng().uniform, low=self.lower_bound, high=self.upper_bound,)
        environments = self.locals['self'].env.envs 
        for env in environments:
            env.set_portion(self.portion_dist)

