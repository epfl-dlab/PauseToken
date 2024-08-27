from stable_baselines3.common.callbacks import BaseCallback  
from lm_stable_baselines.utils import add_filler_tokens
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