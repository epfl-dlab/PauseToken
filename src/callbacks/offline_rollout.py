from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

class OfflineRolloutCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        rollout = pl_module.collect_rollouts()
        