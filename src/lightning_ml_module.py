import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .lr_scheduler import CustomLRScheduler

class LightningLanguageModel(pl.LightningModule):
    def __init__(
        self,
        model,
        step_size,
        n_warmup,
        n_total_iter,
        lr,
        update_every_n_steps,
        strategy: str
    ):
        super().__init__()
        self.model = model
        self.strategy = strategy
        
        #self.scheduler_t_max = scheduler_t_max
        self.step_size = step_size
        self.n_warmup = n_warmup
        self.n_total_iter = n_total_iter
        self.lr = lr
        self.update_every_n_steps = update_every_n_steps
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask).logits

    def configure_optimizers(self):
        if self.strategy == "deepspeed_stage_3":
            optimizer = FusedAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        elif self.strategy == "deepspeed_stage_3_offload":
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.95))

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler_t_max, eta_min=9.65e-7, last_epoch=-1)
        scheduler = CustomLRScheduler(
            optimizer,
            step_size=self.step_size,
            n_warmup=self.n_warmup,
            n_total_iter=self.n_total_iter,
            initial_lr=self.lr,
        )
        sch = [{'scheduler': scheduler, 'interval': 'step', 'frequency': self.update_every_n_steps}]
        return [optimizer], sch

    def training_step(self, batch, batch_idx):
        predictions = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(predictions.transpose(1, 2), batch["labels"], ignore_index=-100)
        perplexity = torch.exp(loss)

        # use loss.item() to convert tensor to python obj which has insanely less mem footprint
        # sync_dist=False will reduce communication across GPUs and only use master gpus loss
        # this is ok since we are anyways shuffling the dataset, so its better to track just one batch loss
        # on_epoch=True will carry the data till the end of epoch which is a lot, so setting it to False
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, sync_dist=False)
        self.log("train_perplexity", perplexity.item(), on_step=True, on_epoch=False, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(predictions.transpose(1, 2), batch["labels"], ignore_index=-100)
        perplexity = torch.exp(loss)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_perplexity", perplexity.item(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def lr_scheduler_step(self, scheduler, metric, _):
        scheduler.step(self.trainer.global_step)

