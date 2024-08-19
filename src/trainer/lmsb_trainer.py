from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from lm_stable_baselines.buffers import LMReplayBuffer
import warnings
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
class LMSBTrainer:
    def __init__(
        self,
        rl_algorithm: BaseAlgorithm,
        inner_loop_timesteps: int,
        n_outer_loops: int,
        learn_callbacks: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        progress_bar: bool = False,
        num_val_samples: int = None,
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
        self.num_val_samples = num_val_samples
        self.logger = self.rl_algorithm.logger
    
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
    
    def evaluation(self, stage: str):
        # Run evaluation on validation set
        #Probably I need to call predict on the model and collect samples
        if self.num_val_samples is None:
            if "val" in self.rl_algorithm.env.envs[0].dataset:
                warnings.warn("num_val_samples was not provided (None), inferring from dataset")
                num_val_samples = len(self.rl_algorithm.env.envs[0].dataset["val"])
                self.num_val_samples = num_val_samples
            else:
                raise ValueError(f"num_val_samples was not provided (None) and no validation samples were found in the dataset so it could not be inferred")
        
        validation_replay_buffer = LMReplayBuffer(
            num_val_samples,
            self.rl_algorithm.observation_space,
            self.rl_algorithm.action_space,
            device = self.rl_algorithm.device,
            n_envs = 1,
            optimize_memory_usage = True,
            **self.rl_algorithm.replay_buffer_kwargs
        )
        
        train_freq = TrainFreq(frequency= num_val_samples, unit = TrainFrequencyUnit.STEP)
        
        rollout = self.rl_algorithm.collect_rollouts(
            self.rl_algorithm.env,
            train_freq= train_freq,
            action_noise=self.rl_algorithm.action_noise,
            learning_starts=0,
            replay_buffer=validation_replay_buffer,
            log_interval=self.learn_kwargs["log_interval"],
            callback=self.learn_kwargs["callback"]
        )
        breakpoint()
        #TODO: Compute or extract metrics (e.g. reward)
        
        #TODO: Save validation metrics
        
        #TODO: Save rollouts to file
    def run_validation(self):
        self.set_stage("val")
        self.evaluation("val")
    
    
    def run_test(self):
        # Run evaluation on test set
        self.eval_dataset.set_stage("test")
        self.evaluation("test")
    
    def save_model(self, save_type = "lm"):
        # Save model
        save_types = ["lm", "rl_alg"]
        assert save_type in save_types, f"Invalid save_type: {save_type}, valid save_types are: {save_types}"
        
        if save_type == "lm":
            self.rl_algorithm.policy.lm.save_pretrained(self.output_dir)
        
        
    def on_validation_start(self):
        pass
        
    def on_validation_end(self):
        self.set_stage("val")
        
    def on_learn_start(self):
        self.set_stage("train")
        
    def on_learn_end(self):
        pass
        
    def on_outer_loop_start(self):
        pass
     
    def fit(self):
       
        for _ in range(self.n_outer_loops):
            self.on_outer_loop_start()
            # Learn
            self.on_learn_start()
            print("Running Learn Stage ... ")
            self.rl_algorithm.learn(**self.learn_kwargs)
            self.on_learn_end()
            
            # Run evaluation on validation set
            
            self.on_validation_start()
            print("Running Validation Stage ... ")
            self.run_validation()
            self.on_validation_end()
            
            print("Saving model")
            # Save model
            self.save_model()
            
        
            