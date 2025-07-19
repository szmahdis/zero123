import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SimpleCheckpointCallback(Callback):
    """Simple checkpoint callback that saves every N steps without monitoring metrics."""
    
    def __init__(self, save_every_n_steps=1000, save_dir="checkpoints", filename_pattern="{epoch:06}-{step:09}"):
        super().__init__()
        self.save_every_n_steps = save_every_n_steps
        self.save_dir = save_dir
        self.filename_pattern = filename_pattern
        self.last_save_step = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Save checkpoint every N training steps."""
        current_step = trainer.global_step
        
        if current_step >= self.save_every_n_steps and current_step - self.last_save_step >= self.save_every_n_steps:
            # Create save directory if it doesn't exist
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Generate filename
            filename = self.filename_pattern.format(
                epoch=trainer.current_epoch,
                step=current_step
            )
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.save_dir, f"{filename}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            
            print(f"Saved checkpoint at step {current_step}: {checkpoint_path}")
            self.last_save_step = current_step