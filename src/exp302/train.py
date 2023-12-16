import argparse
import datetime
import math
import os
import warnings
from glob import glob
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from sklearn.model_selection import KFold
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "302"
COMMENT = """
    kfold, postnorm, high weight decay, long warmup, low s/n threshold, 
    conv transformer, SHAPE positional encoding, bpps bias, efficient impl, 
    param tuning from exp034, swiGLU, split attention ALiBi and bpps, 
    fixed 0-1 clipping, B2T connection option, low grad clipping,
    add error target, stage1 for pseudo label
    """
SEQ_PATH = "../../input/"
train_bpp_paths = glob(f"{SEQ_PATH}/bp_matrix/train*/*.npy", recursive=True)
train_id_to_bpp_paths = {p.split("/")[-1].split(".")[0]: p for p in train_bpp_paths}


class RibonanzaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: np.ndarray,
        mode: str = "train",
        mask_only: bool = False,
        **kwargs,
    ):
        self.mode = mode
        self.seq_map = {"A": 0, "C": 1, "G": 2, "U": 3}
        self.structure_map = {".": 1, "(": 2, ")": 3}  # Add
        df["L"] = df.sequence.apply(len)
        self.Lmax = df["L"].max()
        df_2A3 = df.loc[df.experiment_type == "2A3_MaP"]
        df_DMS = df.loc[df.experiment_type == "DMS_MaP"]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        if mode != "train":
            m = (df_2A3["SN_filter"].values > 0) & (df_DMS["SN_filter"].values > 0)
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)
        else:
            m = ((df_2A3["signal_to_noise"].values > 0.5) & (df_2A3["reads"] > 100)) & (
                (df_DMS["signal_to_noise"].values > 0.5) & (df_DMS["reads"] > 100)
            )
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq_id = df_2A3["sequence_id"].values  # Add
        self.seq = df_2A3["sequence"].values
        self.structure = df_2A3["structure"].values  # Add
        self.L = df_2A3["L"].values

        self.react_2A3 = df_2A3[
            [c for c in df_2A3.columns if "reactivity_0" in c]
        ].values
        self.react_DMS = df_DMS[
            [c for c in df_DMS.columns if "reactivity_0" in c]
        ].values
        self.react_err_2A3 = df_2A3[
            [c for c in df_2A3.columns if "reactivity_error_0" in c]
        ].values
        self.react_err_DMS = df_DMS[
            [c for c in df_DMS.columns if "reactivity_error_0" in c]
        ].values
        self.sn_2A3 = df_2A3["signal_to_noise"].values
        self.sn_DMS = df_DMS["signal_to_noise"].values
        self.mask_only = mask_only

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx: int):
        seq_id = self.seq_id[idx]  # Add
        seq = self.seq[idx]
        structure = self.structure[idx]  # Add
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[: len(seq)] = True
            return {"mask": mask}, {"mask": mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[: len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax - len(seq)))
        bp_matrix = np.load(train_id_to_bpp_paths[seq_id])  # Add
        bp_matrix = np.pad(
            bp_matrix,
            ((0, self.Lmax - len(bp_matrix)), (0, self.Lmax - len(bp_matrix))),
        )  # Add

        structure = [self.structure_map[s] for s in structure]  # Add
        structure = np.array(structure)  # Add
        structure = np.pad(structure, (0, self.Lmax - len(structure)))  # Add
        react_2A3 = self.react_2A3[idx]
        react_DMS = self.react_DMS[idx]
        react_err_2A3 = self.react_err_2A3[idx]
        react_err_DMS = self.react_err_DMS[idx]
        react = np.stack([react_2A3, react_DMS], -1)
        react_err = np.stack([react_err_2A3, react_err_DMS], -1)

        sn = torch.FloatTensor([self.sn_2A3[idx], self.sn_DMS[idx]])

        return {
            "seq": torch.from_numpy(seq),
            "mask": mask,
            "bp_matrix": torch.from_numpy(bp_matrix),  # Add
            "structure": torch.from_numpy(structure),  # Add
        }, {
            "react": torch.from_numpy(react),
            "react_err": torch.from_numpy(react_err),
            "sn": sn,
            "mask": mask,
        }


class LenMatchBatchSampler(BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.dataset[idx]
            if isinstance(s, tuple):
                L = s[0]["mask"].sum()
            else:
                L = s["mask"].sum()
            L = max(1, L // 16)
            if len(buckets[L]) == 0:
                buckets[L] = []
            buckets[L].append(idx)

            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch


def dict_to(x, device="cuda"):
    return {k: x[k].to(device) for k in x}


def to_device(x, device="cuda"):
    return tuple(dict_to(e, device) for e in x)


class RibonanzaDataModule(LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        train_split: np.ndarray,
        val_split: np.ndarray,
        num_workers: int = 4,
        batch_size: int = 16,
        seed: int = 0,
    ):
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size
        self.df = df
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.save_hyperparameters(
            "num_workers",
            "batch_size",
        )

    def create_dataset(self, mode: str = "train") -> RibonanzaDataset:
        if mode == "train":
            return (
                RibonanzaDataset(
                    df=self.df,
                    split=self.train_split,
                    mode=mode,
                    mask_only=False,
                ),
                RibonanzaDataset(
                    df=self.df,
                    split=self.train_split,
                    mode=mode,
                    mask_only=True,
                ),
            )
        else:
            return (
                RibonanzaDataset(
                    df=self.df,
                    split=self.val_split,
                    mode=mode,
                    mask_only=False,
                ),
                RibonanzaDataset(
                    df=self.df,
                    split=self.val_split,
                    mode=mode,
                    mask_only=True,
                ),
            )

    def __dataloader(self, mode: str = "train") -> DataLoader:
        """Train/validation loaders."""
        dataset, dataset_len = self.create_dataset(mode)
        subsampler = DistributedSampler(
            dataset_len, shuffle=(mode == "train"), seed=self.seed
        )
        sampler = LenMatchBatchSampler(
            subsampler,
            batch_size=self._batch_size,
            drop_last=(mode == "train"),
        )
        return DataLoader(
            dataset=dataset,
            num_workers=self._num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            prefetch_factor=10,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="valid")

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="test")

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("RibonanzaDataModule")
        parser.add_argument(
            "--num_workers",
            default=6,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=64,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        return parent_parser

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Conv(nn.Module):
    def __init__(self, d_in, d_out, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        return self.dropout(self.relu(self.bn(self.conv(src))))


class ResidualGraphAttention(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, src, attn):
        h = self.conv2(self.conv1(torch.bmm(src, attn)))
        return self.relu(src + h)
    

class SEResidual(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()
        self.se = SELayer(d_model)

    def forward(self, src):
        h = self.conv2(self.conv1(src))
        return self.se(self.relu(src + h))

class RibonanzaModel(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 9,
        dropout=0.1,
        dropout_res=0.1,
        d_model = 256,
        kernel_size_conv=9, 
        kernel_size_gc=9,
        num_layers=12,
        **kwargs,
    ):
        kernel_sizes = [9,9,9,7,7,7,5,5,5,3,3,3]
        super(RibonanzaModel, self).__init__()
        self.dim = dim
        self.seq_emb = nn.Embedding(4, dim)
        self.conv = Conv(dim, d_model, kernel_size=5, dropout=dropout)
    
        self.blocks = nn.ModuleList([SEResidual(d_model, kernel_size=kernel_sizes[i], dropout=dropout_res) for i in range(num_layers)])
        self.attentions = nn.ModuleList([ResidualGraphAttention(d_model, kernel_size=kernel_sizes[i], dropout=dropout_res) for i in range(num_layers)])
        self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, num_layers=2, bidirectional=True)
        self.proj_out = nn.Linear(d_model, 4)

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x_seq = x0["seq"][:, :Lmax]
        bpps = x0["bp_matrix"][:, :Lmax, :Lmax]
        x = self.seq_emb(x_seq)
        x = x.permute([0, 2, 1])  # [batch, d-emb, seq]
        x = self.conv(x)
        for block, attention in zip(self.blocks, self.attentions):
            x = block(x)
            x = attention(x, bpps)
        x = x.permute([0, 2, 1])  # [batch, seq, features]
        x,_ = self.lstm(x)
        out = self.proj_out(x)
        return out


class RibonanzaLightningModel(pl.LightningModule):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
        lr: float = 1e-3,
        disable_compile: bool = False,
        no_amp: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.no_amp = no_amp
        self.__build_model(
            dim=dim,
            depth=depth,
            head_size=head_size,
            kernel_size=kernel_size,
            b2t_connection=b2t_connection,
        )
        if not disable_compile:
            self.__compile_model()
        self.save_hyperparameters()

    def __build_model(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
    ):
        self.model = RibonanzaModel(
            dim=dim, depth=depth, head_size=head_size, kernel_size=3,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.999)
        self.criterions = {"l1": nn.L1Loss(reduction="none")}

    def __compile_model(self):
        self.model = torch.compile(self.model)
        self.model_ema = torch.compile(self.model_ema)

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        preds = outputs["preds"][:, :, :2]
        preds_err = outputs["preds"][:, :, 2:]
        targets = labels["targets"]
        p = preds[targets["mask"][:, : preds.shape[1]]]
        y = targets["react"][targets["mask"]]
        p_err = preds_err[targets["mask"][:, : preds.shape[1]]]
        y_err = targets["react_err"][targets["mask"]]

        l1_loss = self.criterions["l1"](p, y)
        l1_loss_err = self.criterions["l1"](p_err, y_err)
        if self.training:
            l1_loss = torch.where(
                torch.logical_or(
                    torch.logical_and(p > 10, y > 10),
                    torch.logical_and(p < -10, y < -10),
                ),
                0,
                l1_loss,
            )
        l1_loss = l1_loss[~torch.isnan(l1_loss)].mean()
        l1_loss_err = l1_loss_err[~torch.isnan(l1_loss_err)].mean()
        losses["loss"] = l1_loss + l1_loss_err * 0.2
        losses["l1_loss"] = l1_loss
        losses["l1_loss_err"] = l1_loss_err
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        input, label = batch
        outputs["preds"] = self.model(input)
        loss_target["targets"] = label
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_l1_loss=losses["l1_loss"],
                train_l1_loss_err=losses["l1_loss_err"],
            )
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        input, label = batch
        outputs["preds"] = self.model_ema.module(input).clip(0, 1)
        loss_target["targets"] = label
        loss_target["targets"]["react"][loss_target["targets"]["mask"]] = loss_target[
            "targets"
        ]["react"][loss_target["targets"]["mask"]].clip(0, 1)
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_l1_loss=losses["l1_loss"],
                val_l1_loss_err=losses["l1_loss_err"],
            )
        )
        return step_output

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.05,
                "lr": self.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        optimizer = AdamW(
            self.get_optimizer_parameters(), eps=1e-6 if not self.no_amp else 1e-8
        )
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("RibonanzaLightningModel")
        parser.add_argument(
            "--dim",
            default=192,
            type=int,
            metavar="D",
            dest="dim",
        )
        parser.add_argument(
            "--depth",
            default=12,
            type=int,
            metavar="DPT",
            dest="depth",
        )
        parser.add_argument(
            "--head_size",
            default=32,
            type=int,
            metavar="HS",
            dest="head_size",
        )
        parser.add_argument(
            "--kernel_size",
            default=7,
            type=int,
            metavar="KM",
            dest="kernel_size",
        )
        parser.add_argument(
            "--b2t_connection",
            action="store_true",
            help="b2t_connection option",
            dest="b2t_connection",
        )
        parser.add_argument(
            "--lr",
            default=5e-4,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--disable_compile",
            action="store_true",
            help="disable torch.compile",
            dest="disable_compile",
        )
        return parent_parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    parent_parser.add_argument(
        "--no_amp",
        action="store_true",
        help="not using amp",
        dest="no_amp",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=4, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--epochs", default=30, type=int, metavar="N", help="total number of epochs"
    )
    parser = RibonanzaLightningModel.add_model_specific_args(parent_parser)
    parser = RibonanzaDataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    df = pd.read_parquet(
        os.path.join(SEQ_PATH, "train_data_kmeans_groupkfold_structures.parquet")
    )
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    assert args.fold < 5
    for fold, (train_split, val_split) in enumerate(kf.split(np.arange(len(df) // 2))):
        if args.fold != fold:
            continue
        datamodule = RibonanzaDataModule(
            df=df,
            train_split=train_split,
            val_split=val_split,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model = RibonanzaLightningModel(
            dim=args.dim,
            depth=args.depth,
            head_size=args.head_size,
            kernel_size=args.kernel_size,
            b2t_connection=args.b2t_connection,
            lr=args.lr,
            disable_compile=args.disable_compile,
            no_amp=args.no_amp,
        )

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{fold}"
        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_l1_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_l1_loss", patience=10, log_rank_zero_only=True
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"exp{EXP_ID}/{args.logdir}/fold{fold}",
                save_dir=logdir,
                project="stanford-ribonanza-rna-folding",
                tags=["full_data"],
            )

        trainer = pl.Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=1.0,
            precision="16-mixed" if not args.no_amp else "32-true",
            devices=args.gpus,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            # strategy="ddp",
            max_epochs=args.epochs,
            logger=wandb_logger if not args.debug else True,
            callbacks=[
                loss_checkpoint,
                lr_monitor,
                early_stopping,
            ],
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
            accumulate_grad_batches=64 // args.batch_size
            if args.batch_size < 64
            else 1,
            use_distributed_sampler=False,
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
