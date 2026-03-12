import time
import os

from os import path as os_path
from os import remove as os_remove

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

from stamp.modeling.modeling_approach import ModelingApproach
from stamp.modeling.early_stopping import build_early_stopping
from stamp.modeling.stamp import STAMP


class SupervisedContrastiveLoss(nn.Module):
    """
    Standard SupCon loss (Khosla et al. 2020).
    Expects features shape: (B, V, D) where V is number of views (usually 2).
    Labels shape: (B,)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (B, V, D), L2-normalized recommended
        labels:   (B,)
        """
        device = features.device
        B, V, D = features.shape

        # Flatten views
        feats = features.reshape(B * V, D)  # (B*V, D)
        feats = F.normalize(feats, p=2, dim=-1)

        # Similarity matrix
        logits = torch.matmul(feats, feats.T) / self.temperature  # (B*V, B*V)

        # Mask self-similarity
        self_mask = torch.eye(B * V, device=device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, -1e9)

        # Build positive mask: same class, different sample (and allow different views)
        labels = labels.reshape(B, 1)
        pos_mask_B = torch.eq(labels, labels.T).to(device)  # (B, B)

        # Expand pos mask over views:
        # Each anchor is (i, v), positives are (j, v') where label_i == label_j and (i,v)!=(j,v')
        pos_mask = pos_mask_B.repeat_interleave(V, dim=0).repeat_interleave(V, dim=1)  # (B*V, B*V)
        pos_mask = pos_mask & (~self_mask)

        # For each anchor, compute log prob of positives
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # (B*V, B*V)

        # Average over positives for each anchor
        pos_count = pos_mask.sum(dim=1)  # (B*V,)
        # Avoid division by zero (if a batch has a class appearing once)
        pos_count = torch.clamp(pos_count, min=1)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count  # (B*V,)

        loss = -mean_log_prob_pos.mean()
        return loss


def apply_two_view_augmentations(
    x: torch.Tensor,
    jitter_std: float = 0.01,
    drop_prob: float = 0.0,
    time_mask_prob: float = 0.0,
):
    """
    x: (B, T, S, C)
    Returns x1, x2 with simple augmentations for time-series skeleton-like inputs.
    """
    def aug(inp):
        out = inp

        # Gaussian jitter
        if jitter_std and jitter_std > 0:
            out = out + torch.randn_like(out) * jitter_std

        # Drop random joints (spatial channels)
        if drop_prob and drop_prob > 0:
            B, T, S, C = out.shape
            drop = (torch.rand(B, 1, S, 1, device=out.device) < drop_prob)
            out = out.masked_fill(drop, 0.0)

        # Random time masking (mask some frames)
        if time_mask_prob and time_mask_prob > 0:
            B, T, S, C = out.shape
            tmask = (torch.rand(B, T, 1, 1, device=out.device) < time_mask_prob)
            out = out.masked_fill(tmask, 0.0)

        return out

    return aug(x), aug(x)


class SupConSTAMPModelingApproach(ModelingApproach):
    """
    Stage-1: supervised contrastive pretraining using STAMP's pooled embedding after MHAP (or mean pooling),
    and the projection head (model.project()).

    This mirrors STAMPModelingApproach's structure, but replaces CE/BCE with SupConLoss.
    """

    def __init__(
        self,
        input_dim,
        D,
        n_temporal_channels,
        n_spatial_channels,
        encoder_aggregation,
        use_batch_norm,
        use_instance_norm,
        initial_proj_params,
        final_classifier_params,
        pe_params,
        transformer_params,
        gated_mlp_params,
        mhap_params,
        n_epochs,
        train_batch_size,
        test_batch_size,
        min_epoch,
        early_stopping_params,
        checkpointing_params,
        lr_params,
        optimizer_params,
        problem_type,
        n_classes,
        device,
        # NEW:
        supcon_params,
        aug_params=None,
        n_cls_tokens=None,
        debug_size=None,
        use_tqdm=True,
        store_attention_weights=False,
        use_gradient_clipping=False,
        temporal_channel_selection=None,
        **kwargs
    ):
        super().__init__()

        self.random_seed = None
        self.input_dim = input_dim
        self.D = D
        self.n_temporal_channels = n_temporal_channels
        self.n_spatial_channels = n_spatial_channels
        self.encoder_aggregation = encoder_aggregation
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        self.initial_proj_params = initial_proj_params
        self.final_classifier_params = final_classifier_params
        self.pe_params = pe_params
        self.transformer_params = transformer_params
        self.gated_mlp_params = gated_mlp_params
        self.mhap_params = mhap_params

        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.min_epoch = min_epoch
        self.lr_params = lr_params
        self.optimizer_params = optimizer_params

        self.problem_type = problem_type
        self.n_classes = n_classes
        self.device = torch.device(device)

        self.n_cls_tokens = n_cls_tokens
        self.debug_size = debug_size
        self.use_tqdm = use_tqdm
        self.store_attention_weights = store_attention_weights
        self.use_gradient_clipping = use_gradient_clipping
        self.temporal_channel_selection = temporal_channel_selection
        if self.temporal_channel_selection is not None:
            self.n_temporal_channels = len(self.temporal_channel_selection)

        self.supcon_params = supcon_params or {}
        self.aug_params = aug_params or {}

        # Early stopping
        if early_stopping_params is not None:
            self.use_early_stopping = True
            self.early_stopping = build_early_stopping(early_stopping_params)
            self.tmp_dir = early_stopping_params.get("tmp_dir")
            # We will monitor val_loss (contrastive)
            self.early_stopping_params = early_stopping_params
            # self.early_stopping_params["monitor_metric"] = "val_loss"
        else:
            self.use_early_stopping = False
            self.early_stopping = None
            self.tmp_dir = None

        self.checkpointing_params = checkpointing_params

        # IMPORTANT: STAMP must be constructed with supcon_params so proj_head exists.
        self.model = STAMP(
            use_batch_norm=self.use_batch_norm,
            use_instance_norm=self.use_instance_norm,
            input_dim=self.input_dim,
            D=self.D,
            n_temporal_channels=self.n_temporal_channels,
            n_spatial_channels=self.n_spatial_channels,
            initial_proj_params=self.initial_proj_params,
            pe_params=self.pe_params,
            transformer_params=self.transformer_params,
            gated_mlp_params=self.gated_mlp_params,
            encoder_aggregation=self.encoder_aggregation,
            mhap_params=self.mhap_params,
            final_classifier_params=self.final_classifier_params,
            n_classes=self.n_classes,
            n_cls_tokens=self.n_cls_tokens,
            supcon_params=self.supcon_params.get("proj_head", self.supcon_params),  # flexible naming
        )

        # FLOPs (same pattern as STAMPModelingApproach)
        self.total_flops = None

        self.model.to(self.device)

    def initialize_optimizer(self):
        optimizer_name = self.optimizer_params["optimizer_name"]

        # Default: do NOT train classifier during SupCon stage (sequential option 1)
        train_classifier = bool(self.supcon_params.get("train_classifier", False))
        if not train_classifier and hasattr(self.model, "classifier") and self.model.classifier is not None:
            for p in self.model.classifier.parameters():
                p.requires_grad = False

        params = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer_name == "adam":
            self.model.optimizer = torch.optim.Adam(
                params,
                lr=self.lr_params["initial_lr"],
                betas=self.optimizer_params.get("betas", (0.9, 0.999)),
            )
        elif optimizer_name == "adamw":
            self.model.optimizer = torch.optim.AdamW(
                params,
                lr=self.lr_params["initial_lr"],
                betas=self.optimizer_params.get("betas", (0.9, 0.999)),
                eps=self.optimizer_params.get("eps", 1e-8),
                weight_decay=self.optimizer_params.get("weight_decay", 0.01),
            )
        else:
            raise ValueError("optimizer_name must be 'adam' or 'adamw'")

    def initialize_iterators(self, train_data_loader, val_data_loader):
        if self.use_tqdm:
            return tqdm(train_data_loader, "Training batches..."), tqdm(val_data_loader, "Validation batches...")
        return train_data_loader, val_data_loader

    def train(self, train_data_loader, val_data_loader):
        torch.manual_seed(self.random_seed)

        if self.use_early_stopping:
            self.early_stopping.random_seed = self.random_seed

        temperature = float(self.supcon_params.get("temperature", 0.07))
        criterion = SupervisedContrastiveLoss(temperature=temperature)

        self.initialize_optimizer()

        # Scheduler (copied pattern from STAMPModelingApproach) :contentReference[oaicite:1]{index=1}
        self.scheduler = None
        if self.lr_params.get("use_scheduler", False):
            stype = self.lr_params["scheduler_type"]
            if stype == "one_cycle":
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.model.optimizer,
                    max_lr=self.lr_params["max_lr"],
                    total_steps=self.n_epochs * len(train_data_loader),
                )
            elif stype == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.model.optimizer,
                    T_max=self.n_epochs * len(train_data_loader),
                    eta_min=self.lr_params["eta_min"],
                )
            else:
                raise NotImplementedError(f"Scheduler type {stype} not implemented.")

        self.train_losses = []
        self.val_losses = []
        self.epoch_run_times = []

        train_it, val_it = self.initialize_iterators(train_data_loader, val_data_loader)
        # Checkpointing (best-on-val-loss)
        ckpt_dir = None
        save_best = False
        if self.checkpointing_params is not None:
            ckpt_dir = self.checkpointing_params.get("checkpoint_dir", None)
            save_best = bool(self.checkpointing_params.get("save_best", False))
            if ckpt_dir is not None:
                os.makedirs(ckpt_dir, exist_ok=True)

        best_val_loss = float("inf")
        best_path = os.path.join(ckpt_dir, "best.pth") if (ckpt_dir is not None) else None

        stopped_early = False
        for epoch in range(self.n_epochs):
            print(f"Epoch: {epoch}")
            epoch_start = time.time()

            # -------------------
            # TRAIN
            # -------------------
            self.model.train()
            epoch_train_loss = 0.0

            for seq_batch, label_batch, _sample_key_batch in train_it:
                seq_batch = seq_batch.to(self.device)  # (B,T,S,C)
                label_batch = label_batch.to(self.device).long()

                # Two augmented views
                x1, x2 = apply_two_view_augmentations(
                    seq_batch,
                    jitter_std=float(self.aug_params.get("jitter_std", 0.01)),
                    drop_prob=float(self.aug_params.get("drop_prob", 0.0)),
                    time_mask_prob=float(self.aug_params.get("time_mask_prob", 0.0)),
                )

                self.model.optimizer.zero_grad()

                # We only need projection embeddings
                # Your STAMP.forward(return_proj=True) returns (logits, features, z_proj, attn_or_None)
                _log1, _feat1, z1, _att1 = self.model(x1, return_attention=False, return_proj=True)
                _log2, _feat2, z2, _att2 = self.model(x2, return_attention=False, return_proj=True)

                feats = torch.stack([z1, z2], dim=1)  # (B, 2, proj_dim)
                loss = criterion(feats, label_batch)

                loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                self.model.optimizer.step()

                if self.scheduler is not None and self.lr_params["scheduler_type"] == "one_cycle":
                    self.scheduler.step()

                epoch_train_loss += loss.item()

            if self.scheduler is not None and self.lr_params["scheduler_type"] != "one_cycle":
                self.scheduler.step()

            train_loss = epoch_train_loss / max(1, len(train_data_loader))
            self.train_losses.append(train_loss)

            # -------------------
            # VAL
            # -------------------
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for seq_batch, label_batch, _sample_key_batch in val_it:
                    seq_batch = seq_batch.to(self.device)
                    label_batch = label_batch.to(self.device).long()

                    x1, x2 = apply_two_view_augmentations(
                        seq_batch,
                        jitter_std=float(self.aug_params.get("jitter_std", 0.01)),
                        drop_prob=float(self.aug_params.get("drop_prob", 0.0)),
                        time_mask_prob=float(self.aug_params.get("time_mask_prob", 0.0)),
                    )

                    _log1, _feat1, z1, _ = self.model(x1, return_attention=False, return_proj=True)
                    _log2, _feat2, z2, _ = self.model(x2, return_attention=False, return_proj=True)

                    feats = torch.stack([z1, z2], dim=1)
                    vloss = criterion(feats, label_batch)
                    epoch_val_loss += vloss.item()

            val_loss = epoch_val_loss / max(1, len(val_data_loader))
            self.val_losses.append(val_loss)

            print(f"train_supcon_loss: {train_loss:.4f}, val_supcon_loss: {val_loss:.4f}")
            # Save best checkpoint
            if ckpt_dir is not None and save_best:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    payload = {
                        "epoch": epoch,
                        "val_loss": float(val_loss),
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.model.optimizer.state_dict(),
                        "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler is not None else None),
                        "random_seed": self.random_seed,
                        "supcon_params": self.supcon_params,
                        "aug_params": self.aug_params,
                    }
                    torch.save(payload, best_path)
            # # Early stopping (monitor val_loss)
            # if self.use_early_stopping and epoch > self.min_epoch:
            #     # Some early-stopping implementations expect kwargs; keep it simple:
            #     stopped_early = False
            #     if self.use_early_stopping and epoch > self.min_epoch:
            #         stopped_early = False

            #     if stopped_early:
            #         break

            epoch_end = time.time()
            self.epoch_run_times.append(epoch_end - epoch_start)

        # If early stopping stored a best checkpoint in tmp_dir, reload it (pattern from STAMPModelingApproach)
        # if self.use_early_stopping:
        #     assert self.tmp_dir is not None and os_path.exists(self.tmp_dir), "Tmp dir does not exist."
        #     print(f"Loading best checkpoint from epoch {self.early_stopping.best_epoch}...")
        #     # checkpoint = torch.load(self.tmp_dir + f"/best_checkpoint_seed{self.random_seed}.pth")
        #     # self.model.load_state_dict(checkpoint["model_state_dict"])
        #     del checkpoint
        #     os_remove(self.tmp_dir + f"/best_checkpoint_seed{self.random_seed}.pth")

    def predict(self, test_data_loader):
        """
        SupCon stage-1 doesn't produce meaningful logits unless you trained classifier too.
        For compatibility with the experiment runner, we return dummy preds/probs.

        RECOMMENDED: run stage-2 linear eval using STAMPModelingApproach with frozen backbone.
        """
        self.model.eval()
        test_sample_keys = []
        test_labels = []
        dummy_probs = []
        dummy_preds = []

        with torch.no_grad():
            for _seq_batch, label_batch, sample_key_batch in test_data_loader:
                test_sample_keys.extend(sample_key_batch)
                test_labels.extend(label_batch.numpy().tolist())

                # dummy
                dummy_probs.append(torch.zeros(len(sample_key_batch), 1))
                dummy_preds.append(torch.zeros(len(sample_key_batch), 1))

        prob_df = pd.DataFrame(torch.cat(dummy_probs).numpy(), index=test_sample_keys, columns=["prob"])
        pred_df = pd.DataFrame(torch.cat(dummy_preds).numpy(), index=test_sample_keys, columns=["pred"])

        extra_info = {
            "train_main_losses": self.train_losses,
            "val_main_losses": self.val_losses,
            "train_supcon_losses": self.train_losses,
            "val_supcon_losses": self.val_losses,
            "prob_df": prob_df,
            "test_labels": test_labels,
            "epoch_run_times": self.epoch_run_times,
            "best_epoch": None,

            "train_balanced_acc_list": [],
            "val_balanced_acc_list": [],

            "train_pr_auc_list": [],
            "val_pr_auc_list": [],

            "train_roc_auc_list": [],
            "val_roc_auc_list": [],

            "train_cohen_kappa_list": [],
            "val_cohen_kappa_list": [],

            "train_weighted_f1_list": [],
            "val_weighted_f1_list": [],
        }
        return pred_df, extra_info