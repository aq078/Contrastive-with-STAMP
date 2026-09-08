import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from os import (remove as os_remove, path as os_path)
from tqdm import tqdm
import time
from collections import defaultdict
from fvcore.nn import FlopCountAnalysis
from stamp.modeling.modeling_approach import ModelingApproach
from stamp.modeling.early_stopping import build_early_stopping
from stamp.modeling.utils import calculate_binary_performance_metrics, calculate_multiclass_performance_metrics
from stamp.modeling.stamp import STAMP
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


PENNACTION_ACTION_NAMES = [
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowling",
    "clean_and_jerk",
    "golf_swing",
    "jump_rope",
    "jumping_jacks",
    "pull_ups",
    "push_ups",
    "sit_ups",
    "squats",
    "strumming_guitar",
    "tennis_forehand",
    "tennis_serve",
]

class STAMPModelingApproach(ModelingApproach):
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
        n_cls_tokens=None,
        debug_size=None,
        use_tqdm=True,
        store_attention_weights=False,
        use_gradient_clipping=False,
        label_smoothing=None,
        temporal_channel_selection=None,
        **kwargs
        ):

        super().__init__()

        self.random_seed=None
        self.input_dim = input_dim
        self.D = D
        self.n_temporal_channels = n_temporal_channels
        self.n_spatial_channels = n_spatial_channels
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        self.n_cls_tokens = n_cls_tokens
        self.encoder_aggregation = encoder_aggregation
        self.initial_proj_params = initial_proj_params
        self.final_classifier_params = final_classifier_params
        self.transformer_params = transformer_params
        self.gated_mlp_params = gated_mlp_params
        self.mhap_params = mhap_params
        self.pe_params = pe_params
        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.min_epoch = min_epoch
        self.lr_params = lr_params
        self.optimizer_params = optimizer_params
        self.debug_size = debug_size
        self.use_tqdm = use_tqdm
        self.store_attention_weights = store_attention_weights
        self.use_gradient_clipping = use_gradient_clipping
        self.problem_type = problem_type
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.temporal_channel_selection = temporal_channel_selection
        if self.temporal_channel_selection is not None:
            self.n_temporal_channels = len(self.temporal_channel_selection)
            
        if early_stopping_params is not None:
            self.use_early_stopping = True
            self.early_stopping = build_early_stopping(early_stopping_params)
            self.early_stopping_params = early_stopping_params
            if self.problem_type == 'binary':
                self.early_stopping_params['monitor_metric'] = 'val_roc_auc'
            elif self.problem_type == 'multiclass':
                self.early_stopping_params['monitor_metric'] = 'val_balanced_acc'
            else:
                raise ValueError()
            self.tmp_dir = early_stopping_params.get('tmp_dir')
        else:
            self.use_early_stopping = False
            self.early_stopping = None
            self.tmp_dir = None

        self.checkpointing_params = checkpointing_params

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
        )

        flops = FlopCountAnalysis(self.model, (torch.randn(self.train_batch_size, n_temporal_channels, n_spatial_channels, self.input_dim), False))
        self.total_flops = flops.total()

        self.device = torch.device(device)
        self.model.to(self.device)
        # --- sequential learning (stage-2) ---
        self.load_pretrained_ckpt = kwargs.get("load_pretrained_ckpt", None)
        self.freeze_backbone = kwargs.get("freeze_backbone", False)

        # Delay loading/freezing until train(), because self.random_seed is set after __init__.
        # This lets load_pretrained_ckpt be seed-specific, e.g.
        #   ".../best_seed{seed}.pth"
        self._pretrained_loaded_and_frozen = False
    def _resolve_pretrained_ckpt_path(self):
        """Resolve optional seed-specific pretrained checkpoint path."""
        ckpt_path = self.load_pretrained_ckpt
        if ckpt_path is None:
            return None

        if isinstance(ckpt_path, dict):
            seed = self.random_seed
            return ckpt_path.get(seed, ckpt_path.get(str(seed)))

        if isinstance(ckpt_path, str):
            seed = self.random_seed
            if "{seed}" in ckpt_path or "{random_seed}" in ckpt_path:
                return ckpt_path.format(seed=seed, random_seed=seed)
            return ckpt_path

        raise ValueError(f"Unsupported load_pretrained_ckpt type: {type(ckpt_path)}")

    def _load_pretrained_and_configure_trainable(self):
        """Load checkpoint and apply freeze settings once, after random_seed is known."""
        if self._pretrained_loaded_and_frozen:
            return

        ckpt_path = self._resolve_pretrained_ckpt_path()
        if ckpt_path is not None:
            print(f"Loading pretrained checkpoint: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location=self.device)

            state = ckpt.get("model_state_dict", ckpt)

            missing, unexpected = self.model.load_state_dict(
                state,
                strict=False,
            )

            print(
                f"Loaded pretrained checkpoint. "
                f"Missing keys: {len(missing)}, "
                f"unexpected keys: {len(unexpected)}"
            )
        if self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

            # Stage-2 protocol: train MHAP + classifier on top of frozen backbone.
            if hasattr(self.model, "multi_head_attention_pooling") and self.model.multi_head_attention_pooling is not None:
                for p in self.model.multi_head_attention_pooling.parameters():
                    p.requires_grad = True

            if hasattr(self.model, "classifier") and self.model.classifier is not None:
                for p in self.model.classifier.parameters():
                    p.requires_grad = True
            else:
                raise ValueError("Model has no classifier to train.")

            trainable = [(n, p.numel()) for n, p in self.model.named_parameters() if p.requires_grad]
            print("Trainable params:")
            for n, k in trainable:
                print(n, k)

        self._pretrained_loaded_and_frozen = True

    def train(
        self,
        train_data_loader,
        val_data_loader):

        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)

        # Load seed-specific pretrained checkpoint after random_seed is available.
        self._load_pretrained_and_configure_trainable()

        if self.use_early_stopping:
            self.early_stopping.random_seed = self.random_seed

        # Initialize the criterion
        if self.problem_type == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        elif self.problem_type == 'multiclass':
            #class weight added for sere dataset
            # class_weights = torch.tensor([3.0, 1.0], dtype=torch.float32, device=self.device)
            # criterion = nn.CrossEntropyLoss(
            #     weight=class_weights,
            #     label_smoothing=self.label_smoothing
            # )
            criterion = nn.CrossEntropyLoss(
                            label_smoothing=self.label_smoothing
                        )
        elif self.problem_type == 'regression':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown main loss type: {self.main_loss_type}")

        self.initialize_optimizer()

        # Setup learning rate scheduler
        if self.lr_params['use_scheduler']:
            if self.lr_params['scheduler_type'] == 'one_cycle':
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.model.optimizer,
                    max_lr=self.lr_params['max_lr'],
                    total_steps=self.n_epochs * len(train_data_loader)
                )
            elif self.lr_params['scheduler_type'] == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.model.optimizer,
                    T_max=self.n_epochs * len(train_data_loader),
                    eta_min=self.lr_params['eta_min']
                )
            else:
                raise NotImplementedError(f"Scheduler type {self.lr_params['scheduler_type']} not implemented.")

        # Initialize lists to store metrics
        self.initialize_train_val_performance_lists()
        train_iterator, val_iterator = self.initialize_train_val_iterators(train_data_loader, val_data_loader)

        self.epoch_run_times = []
        stopped_early = False
        for epoch in range(self.n_epochs):  # Number of epochs
            print(f'Epoch: {epoch}')
            epoch_start_time = time.time()
            # Training
            self.model.train()
            epoch_train_main_loss = 0.0
            train_probs = []
            train_preds = []
            train_labels = []
            print('Training...')
            for seq_batch, label_batch, sample_key_batch in train_iterator:

                if self.problem_type == 'binary':
                    label_batch = label_batch.to(torch.float32)
                elif self.problem_type == 'multiclass':
                    label_batch = label_batch.to(torch.long)
                else:
                    pass

                probs, preds, main_loss, _ = self.evaluate_batch(
                    seq_batch=seq_batch,
                    label_batch=label_batch,
                    mode='train',
                    criterion=criterion
                )

                main_loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.model.optimizer.step()

                if self.lr_params['use_scheduler'] and self.lr_params['scheduler_type'] == 'one_cycle':
                    self.scheduler.step()

                if self.problem_type == 'binary':
                    train_probs.append(probs.detach().cpu())
                train_preds.append(preds.detach().cpu())
                train_labels.append(label_batch.detach().cpu())

                # NOTE: We only call .item() after the backward pass because it removes the gradients
                epoch_train_main_loss += main_loss.item()

            if self.lr_params['use_scheduler'] and self.lr_params['scheduler_type'] != 'one_cycle':
                self.scheduler.step()

            train_main_loss = epoch_train_main_loss / len(train_data_loader)

            if self.device.type == 'cuda':
                seq_batch = seq_batch.cpu()
                label_batch = label_batch.cpu()
                probs = probs.cpu() if self.problem_type == 'binary' else None
                main_loss = main_loss.cpu()
                del seq_batch, label_batch, probs
                torch.cuda.empty_cache()

            self.evaluate_split(split_name='train', truths=torch.cat(train_labels), preds=torch.cat(train_preds), probs=torch.cat(train_probs) if self.problem_type == 'binary' else None)
            self.train_main_losses.append(train_main_loss)

            # Validation
            self.model.eval()
            epoch_val_main_loss = 0.0
            val_probs = []
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for seq_batch, label_batch, sample_key_batch in val_iterator:

                    if self.problem_type == 'binary':
                        label_batch = label_batch.to(torch.float32)
                    elif self.problem_type == 'multiclass':
                        label_batch = label_batch.to(torch.long)
                    else:
                        pass

                    probs, preds, main_loss, _ = self.evaluate_batch(
                        seq_batch=seq_batch,
                        label_batch=label_batch,
                        mode='val',
                        criterion=criterion
                    )

                    val_probs.append(probs.cpu())
                    val_preds.append(preds.cpu())
                    val_labels.append(label_batch.cpu())
                    epoch_val_main_loss += main_loss.item()

            val_main_loss = epoch_val_main_loss / len(val_data_loader)

            self.evaluate_split(split_name='val', truths=torch.cat(val_labels), preds=torch.cat(val_preds), probs=torch.cat(val_probs))
            self.val_main_losses.append(val_main_loss)

            print(f'train_main_loss: {train_main_loss:.4f}, val_loss: {val_main_loss:.4f}')
            print(f'train_balanced_acc: {self.train_balanced_acc:.4f}, val_balanced_acc: {self.val_balanced_acc:.4f}')
            if self.problem_type == 'binary':
                print(f'train_pr_auc: {self.train_pr_auc:.4f}, val_pr_auc: {self.val_pr_auc:.4f}')
                print(f'train_roc_auc: {self.train_roc_auc:.4f}, val_roc_auc: {self.val_roc_auc:.4f}')
            elif self.problem_type == 'multiclass':
                print(f'train_cohen_kappa: {self.train_cohen_kappa:.4f}, val_cohen_kappa: {self.val_cohen_kappa:.4f}')
                print(f'train_weighted_f1: {self.train_weighted_f1:.4f}, val_weighted_f1: {self.val_weighted_f1:.4f}')

            print(f'Val CM:\n{self.val_cm}')

            if self.use_early_stopping and epoch > self.min_epoch:
                stopped_early = self.check_early_stopping(epoch, val_loss=val_main_loss, val_roc_auc=self.val_roc_auc, val_balanced_acc=self.val_balanced_acc,
                                                          val_cohen_kappa=self.val_cohen_kappa, val_weighted_f1=self.val_weighted_f1)
                if stopped_early:
                    break

            if self.device.type == 'cuda':
                seq_batch = seq_batch.cpu()
                label_batch = label_batch.cpu()
                probs = probs.cpu()
                main_loss = main_loss.cpu()
                del seq_batch, label_batch, probs, train_probs, train_labels, val_probs, val_labels
                torch.cuda.empty_cache()

            epoch_end_time = time.time()
            self.epoch_run_times.append(epoch_end_time - epoch_start_time)

        if self.use_early_stopping:
            # Load the best model
            assert os_path.exists(self.tmp_dir), 'Tmp dir does not exist.'
            print(f'Loading best checkpoint from epoch {self.early_stopping.best_epoch}...')
            checkpoint = torch.load(self.tmp_dir + f'/best_seed{self.random_seed}.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint

            # Remove the early stopping checkpoint
            os_remove(self.tmp_dir + f'/best_seed{self.random_seed}.pth')

        if self.device.type == 'cuda':
            del train_data_loader, val_data_loader, train_iterator, val_iterator
            torch.cuda.empty_cache()

    def predict(
        self,
        test_data_loader,
    ):
        """
        Run test inference and report both window-level and sequence-level
        diagnostics.

        Expected test batch:
            seq_batch,
            label_batch,
            sample_key_batch,
            sample_metadata_batch

        For sequence-level evaluation, all windows with the same sequence_id
        are grouped and their class-probability vectors are averaged.
        """
        self.model.eval()

        attn_weights_list = []
        test_probs = []
        test_preds = []
        test_sample_keys = []
        test_labels = []
        test_metadata = []
        inference_run_times = []

        with torch.no_grad():
            for batch in test_data_loader:
                if len(batch) == 4:
                    (
                        seq_batch,
                        label_batch,
                        sample_key_batch,
                        sample_metadata_batch,
                    ) = batch
                elif len(batch) == 3:
                    (
                        seq_batch,
                        label_batch,
                        sample_key_batch,
                    ) = batch
                    sample_metadata_batch = None
                else:
                    raise RuntimeError(
                        "Unexpected test batch structure. "
                        f"Expected 3 or 4 items, got {len(batch)}."
                    )

                inference_start_time = time.time()

                probs, preds, _, attn_weights = self.evaluate_batch(
                    seq_batch=seq_batch,
                    label_batch=None,
                    mode="test",
                    criterion=None,
                )

                if (
                    attn_weights is not None
                    and self.store_attention_weights
                ):
                    attn_weights_list.append(
                        attn_weights
                    )

                inference_end_time = time.time()

                inference_run_times.append(
                    inference_end_time
                    - inference_start_time
                )

                test_probs.append(
                    probs.detach().cpu()
                )
                test_preds.append(
                    preds.detach().cpu()
                )
                test_sample_keys.extend(
                    [
                        (
                            key.decode()
                            if isinstance(key, bytes)
                            else str(key)
                        )
                        for key in sample_key_batch
                    ]
                )
                test_labels.extend(
                    label_batch.cpu().numpy().tolist()
                )

                if sample_metadata_batch is None:
                    test_metadata.extend(
                        [None] * len(label_batch)
                    )
                else:
                    if len(sample_metadata_batch) != len(label_batch):
                        raise RuntimeError(
                            "Metadata batch size does not match labels: "
                            f"{len(sample_metadata_batch)} vs "
                            f"{len(label_batch)}"
                        )

                    for metadata in sample_metadata_batch:
                        if metadata is None:
                            test_metadata.append(None)
                        elif isinstance(metadata, dict):
                            test_metadata.append(
                                dict(metadata)
                            )
                        else:
                            raise TypeError(
                                "Expected each metadata item to be dict "
                                f"or None, got "
                                f"{type(metadata).__name__}"
                            )

        if len(attn_weights_list) != 0:
            attn_weights = (
                torch.cat(attn_weights_list)
                .squeeze()
                .cpu()
            )
            attn_weights_dict = {
                sample_id: attn_weights[i]
                for i, sample_id
                in enumerate(test_sample_keys)
            }
        else:
            attn_weights_dict = None

        all_probs_tensor = torch.cat(
            test_probs,
            dim=0,
        )
        all_preds_tensor = torch.cat(
            test_preds,
            dim=0,
        )

        if self.problem_type == "binary":
            prob_df = pd.DataFrame(
                {
                    "prob": (
                        all_probs_tensor
                        .cpu()
                        .numpy()
                    )
                }
            )
            prob_df.index = test_sample_keys
            prob_df.columns = ["prob"]

        elif self.problem_type == "multiclass":
            prob_df = pd.DataFrame(
                all_probs_tensor.cpu().numpy()
            )
            prob_df.index = test_sample_keys
            prob_df.columns = [
                f"prob_class_{i}"
                for i in range(
                    prob_df.shape[1]
                )
            ]
        else:
            prob_df = None

        pred_df = pd.DataFrame(
            {
                "pred": (
                    all_preds_tensor
                    .cpu()
                    .numpy()
                )
            }
        )
        pred_df.index = test_sample_keys
        pred_df.columns = ["pred"]

        test_diagnostics = None
        sequence_diagnostics = None
        sequence_pred_df = None
        sequence_prob_df = None

        if self.problem_type == "multiclass":
            if (
                self.n_classes
                == len(PENNACTION_ACTION_NAMES)
            ):
                action_names = (
                    PENNACTION_ACTION_NAMES
                )
            else:
                action_names = [
                    f"class_{class_idx}"
                    for class_idx
                    in range(self.n_classes)
                ]

            labels = list(
                range(self.n_classes)
            )

            # ====================================================
            # WINDOW-LEVEL DIAGNOSTICS
            # ====================================================
            truths = np.asarray(
                test_labels,
                dtype=np.int64,
            )
            predictions = (
                pred_df["pred"]
                .to_numpy(dtype=np.int64)
            )

            test_cm = confusion_matrix(
                truths,
                predictions,
                labels=labels,
            )

            test_report_text = (
                classification_report(
                    truths,
                    predictions,
                    labels=labels,
                    target_names=action_names,
                    digits=4,
                    zero_division=0,
                )
            )

            test_report_dict = (
                classification_report(
                    truths,
                    predictions,
                    labels=labels,
                    target_names=action_names,
                    output_dict=True,
                    zero_division=0,
                )
            )

            test_accuracy = accuracy_score(
                truths,
                predictions,
            )
            test_balanced_accuracy = (
                balanced_accuracy_score(
                    truths,
                    predictions,
                )
            )
            test_weighted_f1 = f1_score(
                truths,
                predictions,
                average="weighted",
                zero_division=0,
            )
            test_macro_f1 = f1_score(
                truths,
                predictions,
                average="macro",
                zero_division=0,
            )
            test_cohen_kappa = (
                cohen_kappa_score(
                    truths,
                    predictions,
                )
            )

            print()
            print("=" * 70)
            print(
                "STAMP WINDOW-LEVEL TEST DIAGNOSTICS"
            )
            print("=" * 70)
            print(
                f"Accuracy:          "
                f"{test_accuracy:.4f}"
            )
            print(
                f"Balanced accuracy: "
                f"{test_balanced_accuracy:.4f}"
            )
            print(
                f"Weighted F1:       "
                f"{test_weighted_f1:.4f}"
            )
            print(
                f"Macro F1:          "
                f"{test_macro_f1:.4f}"
            )
            print(
                f"Cohen kappa:       "
                f"{test_cohen_kappa:.4f}"
            )

            print(
                "\nWindow-level confusion matrix "
                "(rows=true, columns=predicted):"
            )
            print(test_cm)

            print(
                "\nWindow-level classification "
                "report:\n"
            )
            print(test_report_text)

            window_per_class_rows = []

            for action_name in action_names:
                values = test_report_dict[
                    action_name
                ]

                window_per_class_rows.append(
                    {
                        "action_class": (
                            action_name
                        ),
                        "precision": float(
                            values["precision"]
                        ),
                        "recall": float(
                            values["recall"]
                        ),
                        "f1_score": float(
                            values["f1-score"]
                        ),
                        "support": int(
                            values["support"]
                        ),
                    }
                )

            test_diagnostics = {
                "level": "window",
                "accuracy": float(
                    test_accuracy
                ),
                "balanced_accuracy": float(
                    test_balanced_accuracy
                ),
                "weighted_f1": float(
                    test_weighted_f1
                ),
                "macro_f1": float(
                    test_macro_f1
                ),
                "cohen_kappa": float(
                    test_cohen_kappa
                ),
                "confusion_matrix": test_cm,
                "classification_report": (
                    test_report_dict
                ),
                "per_class": (
                    window_per_class_rows
                ),
            }

            # ====================================================
            # SEQUENCE-LEVEL DIAGNOSTICS
            # ====================================================
            has_metadata = (
                len(test_metadata)
                == len(test_labels)
                and len(test_metadata) > 0
                and all(
                    metadata is not None
                    for metadata in test_metadata
                )
            )

            if has_metadata:
                grouped = defaultdict(
                    lambda: {
                        "probabilities": [],
                        "labels": [],
                        "nframes": [],
                        "sample_keys": [],
                    }
                )

                all_probs_numpy = (
                    all_probs_tensor
                    .cpu()
                    .numpy()
                )

                for (
                    sample_key,
                    label,
                    probabilities,
                    metadata,
                ) in zip(
                    test_sample_keys,
                    test_labels,
                    all_probs_numpy,
                    test_metadata,
                ):
                    sequence_id = metadata.get(
                        "sequence_id"
                    )

                    if sequence_id is None:
                        if "_w" in sample_key:
                            sequence_id = (
                                sample_key.split(
                                    "_w",
                                    1,
                                )[0]
                            )
                        else:
                            raise KeyError(
                                "Cannot recover sequence_id "
                                f"for sample {sample_key!r}"
                            )

                    sequence_id = str(
                        sequence_id
                    )

                    nframes = metadata.get(
                        "nframes"
                    )
                    if nframes is None:
                        nframes = metadata.get(
                            "source_nframes"
                        )

                    if nframes is None:
                        raise KeyError(
                            "Missing nframes/source_nframes "
                            f"for sample {sample_key!r}. "
                            f"Available metadata keys: "
                            f"{sorted(metadata.keys())}"
                        )

                    group = grouped[
                        sequence_id
                    ]
                    group[
                        "probabilities"
                    ].append(
                        np.asarray(
                            probabilities,
                            dtype=np.float64,
                        )
                    )
                    group["labels"].append(
                        int(label)
                    )
                    group["nframes"].append(
                        int(nframes)
                    )
                    group[
                        "sample_keys"
                    ].append(
                        sample_key
                    )

                sequence_rows = []
                sequence_truths = []
                sequence_predictions = []
                sequence_probability_rows = []

                for sequence_id in sorted(
                    grouped
                ):
                    group = grouped[
                        sequence_id
                    ]

                    unique_labels = sorted(
                        set(group["labels"])
                    )
                    if len(unique_labels) != 1:
                        raise RuntimeError(
                            "Inconsistent labels for "
                            f"sequence {sequence_id}: "
                            f"{unique_labels}"
                        )

                    unique_lengths = sorted(
                        set(group["nframes"])
                    )
                    if len(unique_lengths) != 1:
                        raise RuntimeError(
                            "Inconsistent sequence lengths "
                            f"for sequence {sequence_id}: "
                            f"{unique_lengths}"
                        )

                    mean_probabilities = (
                        np.stack(
                            group[
                                "probabilities"
                            ],
                            axis=0,
                        )
                        .mean(axis=0)
                    )

                    true_label = unique_labels[0]
                    predicted_label = int(
                        np.argmax(
                            mean_probabilities
                        )
                    )
                    sequence_length = (
                        unique_lengths[0]
                    )

                    sequence_truths.append(
                        true_label
                    )
                    sequence_predictions.append(
                        predicted_label
                    )

                    sequence_rows.append(
                        {
                            "sequence_id": (
                                sequence_id
                            ),
                            "true_label": (
                                true_label
                            ),
                            "true_class": (
                                action_names[
                                    true_label
                                ]
                            ),
                            "predicted_label": (
                                predicted_label
                            ),
                            "predicted_class": (
                                action_names[
                                    predicted_label
                                ]
                            ),
                            "correct": int(
                                predicted_label
                                == true_label
                            ),
                            "sequence_length": (
                                sequence_length
                            ),
                            "n_windows": len(
                                group[
                                    "probabilities"
                                ]
                            ),
                            "confidence": float(
                                mean_probabilities[
                                    predicted_label
                                ]
                            ),
                        }
                    )

                    sequence_probability_rows.append(
                        mean_probabilities
                    )

                sequence_truths = np.asarray(
                    sequence_truths,
                    dtype=np.int64,
                )
                sequence_predictions = (
                    np.asarray(
                        sequence_predictions,
                        dtype=np.int64,
                    )
                )
                sequence_probability_rows = (
                    np.stack(
                        sequence_probability_rows,
                        axis=0,
                    )
                )

                sequence_cm = confusion_matrix(
                    sequence_truths,
                    sequence_predictions,
                    labels=labels,
                )

                sequence_report_text = (
                    classification_report(
                        sequence_truths,
                        sequence_predictions,
                        labels=labels,
                        target_names=action_names,
                        digits=4,
                        zero_division=0,
                    )
                )

                sequence_report_dict = (
                    classification_report(
                        sequence_truths,
                        sequence_predictions,
                        labels=labels,
                        target_names=action_names,
                        output_dict=True,
                        zero_division=0,
                    )
                )

                sequence_accuracy = (
                    accuracy_score(
                        sequence_truths,
                        sequence_predictions,
                    )
                )
                sequence_balanced_accuracy = (
                    balanced_accuracy_score(
                        sequence_truths,
                        sequence_predictions,
                    )
                )
                sequence_weighted_f1 = (
                    f1_score(
                        sequence_truths,
                        sequence_predictions,
                        average="weighted",
                        zero_division=0,
                    )
                )
                sequence_macro_f1 = f1_score(
                    sequence_truths,
                    sequence_predictions,
                    average="macro",
                    zero_division=0,
                )
                sequence_cohen_kappa = (
                    cohen_kappa_score(
                        sequence_truths,
                        sequence_predictions,
                    )
                )

                print()
                print("=" * 70)
                print(
                    "STAMP SEQUENCE-LEVEL TEST "
                    "DIAGNOSTICS"
                )
                print("=" * 70)
                print(
                    f"Sequences:         "
                    f"{len(sequence_rows)}"
                )
                print(
                    f"Windows:           "
                    f"{len(test_labels)}"
                )
                print(
                    f"Accuracy:          "
                    f"{sequence_accuracy:.4f}"
                )
                print(
                    f"Balanced accuracy: "
                    f"{sequence_balanced_accuracy:.4f}"
                )
                print(
                    f"Weighted F1:       "
                    f"{sequence_weighted_f1:.4f}"
                )
                print(
                    f"Macro F1:          "
                    f"{sequence_macro_f1:.4f}"
                )
                print(
                    f"Cohen kappa:       "
                    f"{sequence_cohen_kappa:.4f}"
                )

                print(
                    "\nSequence-level confusion matrix "
                    "(rows=true, columns=predicted):"
                )
                print(sequence_cm)

                print(
                    "\nSequence-level classification "
                    "report:\n"
                )
                print(sequence_report_text)

                print(
                    "\nSequence-level performance "
                    "by action class:"
                )
                print(
                    f"{'Action':22s} "
                    f"{'Precision':>10s} "
                    f"{'Recall':>10s} "
                    f"{'F1':>10s} "
                    f"{'Support':>8s}"
                )
                print("-" * 66)

                sequence_per_class_rows = []

                for action_name in action_names:
                    values = (
                        sequence_report_dict[
                            action_name
                        ]
                    )

                    row = {
                        "action_class": (
                            action_name
                        ),
                        "precision": float(
                            values["precision"]
                        ),
                        "recall": float(
                            values["recall"]
                        ),
                        "f1_score": float(
                            values["f1-score"]
                        ),
                        "support": int(
                            values["support"]
                        ),
                    }
                    sequence_per_class_rows.append(
                        row
                    )

                    print(
                        f"{action_name:22s} "
                        f"{row['precision']:10.4f} "
                        f"{row['recall']:10.4f} "
                        f"{row['f1_score']:10.4f} "
                        f"{row['support']:8d}"
                    )

                # ====================================================
                # SEQUENCE-LENGTH DIAGNOSTICS
                # ====================================================
                sequence_length_df = pd.DataFrame(sequence_rows)

                print()
                print("=" * 70)
                print("STAMP SEQUENCE-LENGTH DIAGNOSTICS")
                print("=" * 70)

                correct_df = sequence_length_df[
                    sequence_length_df["correct"] == 1
                ]
                wrong_df = sequence_length_df[
                    sequence_length_df["correct"] == 0
                ]

                print(f"Correct sequences: {len(correct_df)}")
                print(f"Wrong sequences:   {len(wrong_df)}")
                print()
                print(
                    f"{'Group':18s} "
                    f"{'Mean':>8s} "
                    f"{'Median':>8s} "
                    f"{'Std':>8s} "
                    f"{'Min':>8s} "
                    f"{'Max':>8s}"
                )
                print("-" * 66)

                length_summary_rows = []

                for group_name, group_df in (
                    ("All", sequence_length_df),
                    ("Correct", correct_df),
                    ("Wrong", wrong_df),
                ):
                    lengths = group_df[
                        "sequence_length"
                    ].to_numpy(dtype=np.float64)

                    if len(lengths) == 0:
                        summary_row = {
                            "group": group_name,
                            "count": 0,
                            "mean": np.nan,
                            "median": np.nan,
                            "std": np.nan,
                            "min": np.nan,
                            "max": np.nan,
                        }
                    else:
                        summary_row = {
                            "group": group_name,
                            "count": int(len(lengths)),
                            "mean": float(np.mean(lengths)),
                            "median": float(np.median(lengths)),
                            "std": float(np.std(lengths)),
                            "min": int(np.min(lengths)),
                            "max": int(np.max(lengths)),
                        }

                    length_summary_rows.append(summary_row)

                    if summary_row["count"] == 0:
                        print(
                            f"{group_name:18s} "
                            f"{'n/a':>8s} "
                            f"{'n/a':>8s} "
                            f"{'n/a':>8s} "
                            f"{'n/a':>8s} "
                            f"{'n/a':>8s}"
                        )
                    else:
                        print(
                            f"{group_name:18s} "
                            f"{summary_row['mean']:8.1f} "
                            f"{summary_row['median']:8.1f} "
                            f"{summary_row['std']:8.1f} "
                            f"{summary_row['min']:8d} "
                            f"{summary_row['max']:8d}"
                        )

                length_bin_edges = [
                    0, 32, 64, 96, 128, 160,
                    192, 256, 512, np.inf,
                ]
                length_bin_names = [
                    "0-31",
                    "32-63",
                    "64-95",
                    "96-127",
                    "128-159",
                    "160-191",
                    "192-255",
                    "256-511",
                    "512+",
                ]

                sequence_length_df["length_bin"] = pd.cut(
                    sequence_length_df["sequence_length"],
                    bins=length_bin_edges,
                    labels=length_bin_names,
                    include_lowest=True,
                    right=False,
                )

                print()
                print(
                    "Sequence-level accuracy by original "
                    "sequence length:"
                )
                print(
                    f"{'Length bin':12s} "
                    f"{'N':>6s} "
                    f"{'Accuracy':>10s} "
                    f"{'Mean len':>10s}"
                )
                print("-" * 44)

                length_bin_rows = []

                for length_bin in length_bin_names:
                    subset = sequence_length_df[
                        sequence_length_df["length_bin"]
                        == length_bin
                    ]

                    if len(subset) == 0:
                        continue

                    bin_row = {
                        "length_bin": length_bin,
                        "support": int(len(subset)),
                        "accuracy": float(
                            subset["correct"].mean()
                        ),
                        "mean_sequence_length": float(
                            subset["sequence_length"].mean()
                        ),
                    }
                    length_bin_rows.append(bin_row)

                    print(
                        f"{length_bin:12s} "
                        f"{bin_row['support']:6d} "
                        f"{bin_row['accuracy']:10.4f} "
                        f"{bin_row['mean_sequence_length']:10.1f}"
                    )

                print()
                print(
                    "Sequence length and accuracy "
                    "by action class:"
                )
                print(
                    f"{'Action':22s} "
                    f"{'N':>5s} "
                    f"{'Accuracy':>10s} "
                    f"{'Mean len':>10s} "
                    f"{'Median':>10s} "
                    f"{'Correct len':>12s} "
                    f"{'Wrong len':>10s}"
                )
                print("-" * 88)

                action_length_rows = []

                for class_idx, action_name in enumerate(action_names):
                    action_df = sequence_length_df[
                        sequence_length_df["true_label"]
                        == class_idx
                    ]
                    action_correct_df = action_df[
                        action_df["correct"] == 1
                    ]
                    action_wrong_df = action_df[
                        action_df["correct"] == 0
                    ]

                    support = int(len(action_df))
                    if support == 0:
                        continue

                    action_row = {
                        "action_class": action_name,
                        "support": support,
                        "accuracy": float(
                            action_df["correct"].mean()
                        ),
                        "mean_sequence_length": float(
                            action_df["sequence_length"].mean()
                        ),
                        "median_sequence_length": float(
                            action_df["sequence_length"].median()
                        ),
                        "correct_mean_length": (
                            float(
                                action_correct_df[
                                    "sequence_length"
                                ].mean()
                            )
                            if len(action_correct_df) > 0
                            else np.nan
                        ),
                        "incorrect_mean_length": (
                            float(
                                action_wrong_df[
                                    "sequence_length"
                                ].mean()
                            )
                            if len(action_wrong_df) > 0
                            else np.nan
                        ),
                    }
                    action_length_rows.append(action_row)

                    correct_len_text = (
                        f"{action_row['correct_mean_length']:.1f}"
                        if not np.isnan(
                            action_row["correct_mean_length"]
                        )
                        else "n/a"
                    )
                    wrong_len_text = (
                        f"{action_row['incorrect_mean_length']:.1f}"
                        if not np.isnan(
                            action_row["incorrect_mean_length"]
                        )
                        else "n/a"
                    )

                    print(
                        f"{action_name:22s} "
                        f"{support:5d} "
                        f"{action_row['accuracy']:10.4f} "
                        f"{action_row['mean_sequence_length']:10.1f} "
                        f"{action_row['median_sequence_length']:10.1f} "
                        f"{correct_len_text:>12s} "
                        f"{wrong_len_text:>10s}"
                    )

                sequence_pred_df = (
                    sequence_length_df
                    .set_index("sequence_id")
                )

                sequence_prob_df = pd.DataFrame(
                    sequence_probability_rows,
                    index=sequence_pred_df.index,
                    columns=[
                        f"prob_class_{i}"
                        for i in range(
                            self.n_classes
                        )
                    ],
                )

                sequence_diagnostics = {
                    "level": "sequence",
                    "aggregation": (
                        "mean_window_probabilities"
                    ),
                    "n_sequences": int(
                        len(sequence_rows)
                    ),
                    "n_windows": int(
                        len(test_labels)
                    ),
                    "accuracy": float(
                        sequence_accuracy
                    ),
                    "balanced_accuracy": float(
                        sequence_balanced_accuracy
                    ),
                    "weighted_f1": float(
                        sequence_weighted_f1
                    ),
                    "macro_f1": float(
                        sequence_macro_f1
                    ),
                    "cohen_kappa": float(
                        sequence_cohen_kappa
                    ),
                    "confusion_matrix": (
                        sequence_cm
                    ),
                    "classification_report": (
                        sequence_report_dict
                    ),
                    "per_class": (
                        sequence_per_class_rows
                    ),
                    "sequence_predictions": (
                        sequence_pred_df
                    ),
                    "sequence_probabilities": (
                        sequence_prob_df
                    ),
                    "length_summary": (
                        length_summary_rows
                    ),
                    "length_bins": (
                        length_bin_rows
                    ),
                    "action_length_diagnostics": (
                        action_length_rows
                    ),
                }

            else:
                print()
                print(
                    "WARNING: sequence-level diagnostics "
                    "were skipped because complete test "
                    "metadata was not provided."
                )

        extra_info = {
            "best_epoch": (
                self.early_stopping.best_epoch
                if self.use_early_stopping
                else None
            ),
            "train_main_losses": (
                self.train_main_losses
            ),
            "train_balanced_acc_list": (
                self.train_balanced_acc_list
            ),
            "train_roc_auc_list": (
                self.train_roc_auc_list
            ),
            "train_pr_auc_list": (
                self.train_pr_auc_list
            ),
            "train_cohen_kappa_list": (
                self.train_cohen_kappa_list
            ),
            "train_weighted_f1_list": (
                self.train_weighted_f1_list
            ),
            "train_cm_list": (
                self.train_cm_list
            ),
            "val_main_losses": (
                self.val_main_losses
            ),
            "val_balanced_acc_list": (
                self.val_balanced_acc_list
            ),
            "val_roc_auc_list": (
                self.val_roc_auc_list
            ),
            "val_pr_auc_list": (
                self.val_pr_auc_list
            ),
            "val_cohen_kappa_list": (
                self.val_cohen_kappa_list
            ),
            "val_weighted_f1_list": (
                self.val_weighted_f1_list
            ),
            "val_cm_list": (
                self.val_cm_list
            ),
            "attn_weights": (
                attn_weights_dict
            ),
            "prob_df": prob_df,
            "test_labels": test_labels,
            "test_metadata": test_metadata,
            "test_diagnostics": (
                test_diagnostics
            ),
            "sequence_diagnostics": (
                sequence_diagnostics
            ),
            "sequence_pred_df": (
                sequence_pred_df
            ),
            "sequence_prob_df": (
                sequence_prob_df
            ),
            "epoch_run_times": (
                self.epoch_run_times
            ),
            "inference_run_times": (
                inference_run_times
            ),
        }

        return pred_df, extra_info

    def initialize_optimizer(self):
        optimizer_name = self.optimizer_params['optimizer_name']
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("No trainable parameters found (did you freeze everything?).")
        if optimizer_name == 'adam':
            self.model.optimizer = torch.optim.Adam(params, lr=self.lr_params['initial_lr'], betas=self.optimizer_params.get('betas', (0.9, 0.999)))
        elif optimizer_name == 'adamw':
            self.model.optimizer = torch.optim.AdamW(
                params,
                lr=self.lr_params['initial_lr'],
                betas=self.optimizer_params.get('betas', (0.9, 0.999)),
                eps=self.optimizer_params.get('eps', 1e-8),
                weight_decay=self.optimizer_params.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f'Given optimizer name, {optimizer_name}, is not valid. Valid names are adam and adamw.')

    def initialize_train_val_performance_lists(self):
        self.train_main_losses = []
        self.train_balanced_acc_list = []
        self.train_pr_auc_list = []
        self.train_roc_auc_list = []
        self.train_cohen_kappa_list = []
        self.train_weighted_f1_list = []
        self.train_cm_list = []

        self.val_main_losses = []
        self.val_pr_auc_list = []
        self.val_roc_auc_list = []
        self.val_balanced_acc_list = []
        self.val_cohen_kappa_list = []
        self.val_weighted_f1_list = []
        self.val_cm_list = []

    def initialize_train_val_iterators(self, train_data_loader, val_data_loader):
        if self.use_tqdm:
            train_iterator = tqdm(train_data_loader, 'Training batches...')
            val_iterator = tqdm(val_data_loader, 'Validation batches...')
        else:
            train_iterator = train_data_loader
            val_iterator = val_data_loader

        return train_iterator, val_iterator

    def evaluate_batch(
        self,
        seq_batch,
        label_batch,
        mode,
        criterion
        ):
        # Move tensors to the specified device
        seq_batch = seq_batch.to(self.device) # Shape: (batch_size, max_hr, n_channels, n_features)

        if mode == 'train':
            self.model.optimizer.zero_grad()
        return_attention = (mode == 'test' and self.store_attention_weights)
        logits, attn_weights = self.model(x=seq_batch, return_attention=return_attention) # Binary shape: (batch_size, 1), Multiclass shape: (batch_size, n_classes)
        if self.problem_type == 'binary':
            logits = logits.squeeze()  # Remove class dimension for binary

        # Make sure each tensor has atleast 1 dim to prevent error
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)

        if criterion is not None:
            label_batch = label_batch.to(self.device) # Shape: (batch_size)
            loss = criterion(logits, label_batch) # Single value
        else:
            loss = None

        # Run outputs through sigmoid to get probabilities
        if self.problem_type == 'binary':
            probs = torch.sigmoid(logits)
            preds = torch.gt(probs, 0.5).long()
        elif self.problem_type == 'multiclass':
            probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
            preds = torch.argmax(logits, dim=-1)   # Get predicted class indices

        return probs, preds, loss, attn_weights

    def evaluate_split(self, split_name, truths, preds, probs=None):
        """
        Evaluate metrics for a given split (train/val/test) and update lists.
        Args:
            split_name (str): 'train', 'val', or 'test'
            truths (array-like): Ground-truth labels
            preds (array-like): Predicted labels
            probs (array-like or None): Predicted probabilities (binary only)
        """
        if self.problem_type == 'binary':
            balanced_acc, pr_auc, roc_auc, cm = calculate_binary_performance_metrics(
                truths=truths,
                probs=probs,
                preds=preds
            )

            # Dynamically choose which lists to update
            getattr(self, f"{split_name}_pr_auc_list").append(pr_auc)
            getattr(self, f"{split_name}_roc_auc_list").append(roc_auc)

            setattr(self, f"{split_name}_pr_auc", pr_auc)
            setattr(self, f"{split_name}_roc_auc", roc_auc)
            setattr(self, f"{split_name}_cohen_kappa", None)
            setattr(self, f"{split_name}_weighted_f1", None)

        elif self.problem_type == 'multiclass':
            balanced_acc, cohen_kappa, weighted_f1, cm = calculate_multiclass_performance_metrics(
                truths=truths,
                preds=preds
            )

            getattr(self, f"{split_name}_cohen_kappa_list").append(cohen_kappa)
            getattr(self, f"{split_name}_weighted_f1_list").append(weighted_f1)

            setattr(self, f"{split_name}_pr_auc", None)
            setattr(self, f"{split_name}_roc_auc", None)
            setattr(self, f"{split_name}_cohen_kappa", cohen_kappa)
            setattr(self, f"{split_name}_weighted_f1", weighted_f1)

        else:
            raise ValueError(f"Invalid problem_type: {self.problem_type}")

        getattr(self, f"{split_name}_balanced_acc_list").append(balanced_acc)
        getattr(self, f"{split_name}_cm_list").append(cm)

        setattr(self, f"{split_name}_balanced_acc", balanced_acc)
        setattr(self, f"{split_name}_cm", cm)