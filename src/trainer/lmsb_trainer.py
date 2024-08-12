from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from datasets import Dataset
import warnings
class LMSBTrainer:
    def __init__(
        self,
        rl_algorithm: BaseAlgorithm,
        eval_dataset: Dataset,
        inner_loop_timesteps: int,
        n_outer_loops: int,
        learn_callbacks: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        progress_bar: bool = False,
        callbacks = None #TODO: define type
    ):
        self.learn_kwargs = {
            "total_timesteps": inner_loop_timesteps,
            "callback": learn_callbacks,
            "log_interval": log_interval,
            "tb_log_name": tb_log_name,
            "progress_bar": progress_bar
        }
        self.rl_algorithm = rl_algorithm
        self.n_outer_loops = n_outer_loops
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks
    
    def set_stage(self, stage: str):
        valid_stages = ["train", "val", "test"]
        assert stage in valid_stages, f"Invalid stage: {stage}, valid stages are: {valid_stages}"
        self.stage = stage        
        
        if stage == "train":
            read_sequentially = False
            self.rl_algorithm.policy.train()
        else:
            read_sequentially = True
            self.rl_algorithm.policy.eval()
        
        self.rl_algorithm.env.set_stage(stage, read_sequentially = read_sequentially)
    
    def run_validation(self):
        # Run evaluation on validation set
        #Probably I need to call predict on the model and collect samples
        pass
    
    def run_test(self):
        # Run evaluation on test set
        pass
    
    def save_model(self):
        # Save model
        pass
    
    def call_callback(self, callback_name: str):
        callback_fn = getattr(self.callbacks, callback_name, None)
        if callable(callback_fn):
            callback_fn(self)
        else:
            warnings.warn(f"Callback {callback_name} not found or not callable, This should not happen. Skipping...")
    
    def fit(self):
        for _ in range(self.n_outer_loops):
            self.call_callback("on_outer_loop_start")
            # Learn
            self.set_stage("train")
            self.call_callback("on_learn_start")
            self.rl_algorithm.learn(**self.learn_kwargs)
            self.call_callback("on_learn_end")
            
            # Run evaluation on validation set
            self.set_stage("val")
            self.call_callback("on_validation_start")
            self.run_validation()
            self.call_callback("on_validation_end")
            
            # Save model
            self.save_model()
            