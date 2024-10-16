from stable_baselines3.common.callbacks import BaseCallback  
from lm_stable_baselines.utils import add_filler_tokens
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
        self.prev_num_steps = 0
        
    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        if self.prev_num_steps != self.locals["num_collected_episodes"]:
            num_steps = self.locals["num_collected_steps"]
            self.pbar.update(num_steps - self.prev_num_steps)
            self.prev_num_steps = num_steps
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()