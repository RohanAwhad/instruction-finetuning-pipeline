import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

class SFTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        eos_token_id,
        train_batch_size,
        val_dataset=None,
        val_batch_size=None,
        num_workers=0,
        prefetch_factor=1000,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.eos_token_id = eos_token_id
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        if self.train_dataset: self.train_dataloader = self._train_dataloader
        if self.val_dataset: self.val_dataloader = self._val_dataloader

    def collate_fn(self, batch):
        # get the max length of the sequences in the batch
        max_len = max([len(item["input_ids"]) for item in batch])

        # pad the sequences
        for item in batch:
            item["input_ids"] = torch.cat(
                [
                    item["input_ids"],
                    torch.tensor([self.eos_token_id] * (max_len - len(item["input_ids"]))),
                ],
                dim=0,
            )
            item["attention_mask"] = torch.cat(
                [
                    item["attention_mask"],
                    torch.tensor([0] * (max_len - len(item["attention_mask"]))),
                ],
                dim=0,
            )
            item["labels"] = torch.cat(
                [
                    item["labels"],
                    torch.tensor([-100] * (max_len - len(item["labels"]))),
                ],
                dim=0,
            )

        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]).int(),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]).int(),
            "labels": torch.stack([item["labels"] for item in batch]).long(),
        }

    def _train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),  # recommended by PL. Blocks the CPU even when not using, but speeds up training. So half CPUs are for train and other half for val
            pin_memory=True,  # (ref): https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
            prefetch_factor=self.prefetch_factor,  # prefetch algo: prefetch_factor * num_workers (samples and not batches)
        )

    def _val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            pin_memory=True,
            prefetch_factor=self.val_batch_size * 50,
        )
