"""ViSoBERT baseline and controlled global/local ablation heads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.modeling_outputs import SequenceClassifierOutput
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - dependency bootstrap path
    torch = None
    nn = None
    F = None
    AutoConfig = AutoModel = AutoModelForSequenceClassification = None
    PretrainedConfig = object
    PreTrainedModel = object
    SequenceClassifierOutput = None
    _IMPORT_ERROR = exc


CUSTOM_VARIANTS = ("mean", "cnn", "concat", "gated")
CNN_POOLING_MODES = ("max", "max_mean")
LOSS_TYPES = ("cross_entropy", "focal")
LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def require_model_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError("Install torch and transformers to use model code") from _IMPORT_ERROR


def _config_from_dict(value: Mapping):
    payload = deepcopy(dict(value))
    model_type = payload.pop("model_type", None)
    if not model_type:
        raise ValueError("encoder_config must contain model_type")
    return AutoConfig.for_model(model_type, **payload)


if _IMPORT_ERROR is None:

    class GlobalLocalConfig(PretrainedConfig):
        """Serializable composition config for custom ViSoBERT heads."""

        model_type = "visobert-global-local"
        is_composition = True

        def __init__(
            self,
            *,
            encoder_config: Mapping | None = None,
            encoder_model_id: str = "uitnlp/visobert",
            encoder_revision: str | None = None,
            variant: str = "gated",
            cnn_kernels: Sequence[int] = (2, 3, 4),
            cnn_channels: int = 64,
            fusion_dim: int = 256,
            head_dropout: float = 0.1,
            cnn_pooling: str = "max",
            normalize_branches: bool = False,
            residual_gate: bool = False,
            loss_type: str = "cross_entropy",
            class_weights: Sequence[float] | None = None,
            focal_gamma: float = 0.0,
            num_labels: int = 3,
            **kwargs,
        ):
            if variant not in CUSTOM_VARIANTS:
                raise ValueError(f"variant must be one of {CUSTOM_VARIANTS}")
            if encoder_config is None:
                # Allows config inspection, but construction of a model will fail
                # until a real encoder config is supplied.
                encoder_config = {}
            kernels = tuple(int(value) for value in cnn_kernels)
            if not kernels or any(value < 1 for value in kernels):
                raise ValueError("cnn_kernels must contain positive integers")
            if cnn_channels < 1 or fusion_dim < 1:
                raise ValueError("CNN/fusion dimensions must be positive")
            if not 0 <= head_dropout < 1:
                raise ValueError("head_dropout must be in [0,1)")
            if cnn_pooling not in CNN_POOLING_MODES:
                raise ValueError(f"cnn_pooling must be one of {CNN_POOLING_MODES}")
            if loss_type not in LOSS_TYPES:
                raise ValueError(f"loss_type must be one of {LOSS_TYPES}")
            if focal_gamma < 0:
                raise ValueError("focal_gamma must be nonnegative")
            weights = None if class_weights is None else [float(value) for value in class_weights]
            if weights is not None and (
                len(weights) != num_labels or any(value <= 0 for value in weights)
            ):
                raise ValueError("class_weights must contain one positive value per label")
            super().__init__(num_labels=num_labels, **kwargs)
            self.encoder_config = dict(encoder_config)
            self.encoder_model_id = encoder_model_id
            self.encoder_revision = encoder_revision
            self.variant = variant
            self.cnn_kernels = list(kernels)
            self.cnn_channels = int(cnn_channels)
            self.fusion_dim = int(fusion_dim)
            self.head_dropout = float(head_dropout)
            self.cnn_pooling = cnn_pooling
            self.normalize_branches = bool(normalize_branches)
            self.residual_gate = bool(residual_gate)
            self.loss_type = loss_type
            self.class_weights = weights
            self.focal_gamma = float(focal_gamma)


    def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


    class MaskedCnnPool(nn.Module):
        """Multi-kernel CNN with padding-safe center-position max pooling."""

        def __init__(
            self,
            hidden_size: int,
            channels: int,
            kernels: Sequence[int],
            pooling: str = "max",
        ):
            super().__init__()
            if pooling not in CNN_POOLING_MODES:
                raise ValueError(f"pooling must be one of {CNN_POOLING_MODES}")
            self.kernels = tuple(int(value) for value in kernels)
            self.pooling = pooling
            self.convs = nn.ModuleList([
                nn.Conv1d(hidden_size, channels, kernel_size=value, padding=0)
                for value in self.kernels
            ])

        @property
        def output_size(self) -> int:
            multiplier = 2 if self.pooling == "max_mean" else 1
            return multiplier * sum(conv.out_channels for conv in self.convs)

        def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            if hidden_states.ndim != 3 or attention_mask.ndim != 2:
                raise ValueError("Expected hidden [B,L,H] and mask [B,L]")
            if hidden_states.shape[:2] != attention_mask.shape:
                raise ValueError("Hidden states and attention mask shapes disagree")
            if not bool(attention_mask.bool().any(dim=1).all()):
                raise ValueError("Every sample must have at least one unmasked token")
            x = (
                hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            ).transpose(1, 2)
            pooled = []
            for conv in self.convs:
                kernel = int(conv.kernel_size[0])
                left = (kernel - 1) // 2
                right = kernel - 1 - left
                features = F.gelu(conv(F.pad(x, (left, right))))
                features = features.masked_fill(
                    ~attention_mask[:, None, :].bool(),
                    torch.finfo(features.dtype).min,
                )
                maximum = features.amax(dim=-1)
                if self.pooling == "max_mean":
                    valid = attention_mask[:, None, :].to(features.dtype)
                    mean = features.masked_fill(~attention_mask[:, None, :].bool(), 0).sum(
                        dim=-1
                    ) / valid.sum(dim=-1).clamp_min(1)
                    pooled.extend((maximum, mean))
                else:
                    pooled.append(maximum)
            return torch.cat(pooled, dim=-1)


    def balanced_class_weights(
        labels: Sequence[int], *, num_labels: int = 3, power: float = 0.5
    ) -> list[float]:
        """Return normalized inverse-frequency weights without using dev/test data."""
        if not 0 <= power <= 1:
            raise ValueError("power must be in [0,1]")
        counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_labels)
        if counts.numel() != num_labels or bool((counts == 0).any()):
            raise ValueError("Every class must occur in the training labels")
        weights = counts.to(torch.float64).pow(-power)
        weights = weights / weights.mean()
        return [float(value) for value in weights]


    def focal_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        class_weights: torch.Tensor | None = None,
        gamma: float = 1.5,
    ) -> torch.Tensor:
        """Multi-class focal loss with optional training-only class weights."""
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        per_sample = F.cross_entropy(
            logits, labels, weight=class_weights, reduction="none"
        )
        true_probability = logits.softmax(dim=-1).gather(1, labels[:, None]).squeeze(1)
        return (((1.0 - true_probability).clamp_min(0) ** gamma) * per_sample).mean()


    class ViSoBERTGlobalLocalForSequenceClassification(PreTrainedModel):
        config_class = GlobalLocalConfig
        base_model_prefix = "encoder"
        main_input_name = "input_ids"
        accepts_loss_kwargs = False

        def __init__(self, config: GlobalLocalConfig):
            super().__init__(config)
            if not config.encoder_config:
                raise ValueError("encoder_config is required to construct the model")
            encoder_config = _config_from_dict(config.encoder_config)
            self.encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
            hidden_size = int(encoder_config.hidden_size)
            self.dropout = nn.Dropout(config.head_dropout)

            if config.variant == "mean":
                self.classifier = nn.Linear(hidden_size, config.num_labels)
            else:
                self.local_pool = MaskedCnnPool(
                    hidden_size,
                    config.cnn_channels,
                    config.cnn_kernels,
                    pooling=config.cnn_pooling,
                )
                self.local_projection = nn.Linear(
                    self.local_pool.output_size, config.fusion_dim
                )
                if config.variant == "cnn":
                    self.classifier = nn.Linear(config.fusion_dim, config.num_labels)
                else:
                    self.global_projection = nn.Linear(hidden_size, config.fusion_dim)
                    if config.variant == "concat":
                        self.classifier = nn.Linear(2 * config.fusion_dim, config.num_labels)
                    else:
                        self.gate = nn.Linear(2 * config.fusion_dim, config.fusion_dim)
                        if config.residual_gate:
                            self.fusion_norm = nn.LayerNorm(config.fusion_dim)
                        self.classifier = nn.Linear(config.fusion_dim, config.num_labels)
                if config.normalize_branches:
                    self.local_norm = nn.LayerNorm(config.fusion_dim)
                    if config.variant in {"concat", "gated"}:
                        self.global_norm = nn.LayerNorm(config.fusion_dim)
            weights = [] if config.class_weights is None else config.class_weights
            self.register_buffer(
                "class_weight_tensor",
                torch.tensor(weights, dtype=torch.float32),
                persistent=False,
            )
            self.post_init()

        @classmethod
        def from_encoder_pretrained(
            cls,
            model_id: str,
            *,
            revision: str,
            token: str | None,
            variant: str,
            cnn_kernels: Sequence[int] = (2, 3, 4),
            cnn_channels: int = 64,
            fusion_dim: int = 256,
            head_dropout: float = 0.1,
            cnn_pooling: str = "max",
            normalize_branches: bool = False,
            residual_gate: bool = False,
            loss_type: str = "cross_entropy",
            class_weights: Sequence[float] | None = None,
            focal_gamma: float = 0.0,
            attn_implementation: str = "eager",
        ):
            if not revision:
                raise ValueError("A locked model revision is required")
            encoder_config = AutoConfig.from_pretrained(
                model_id, revision=revision, token=token
            )
            config = GlobalLocalConfig(
                encoder_config=encoder_config.to_dict(),
                encoder_model_id=model_id,
                encoder_revision=revision,
                variant=variant,
                cnn_kernels=cnn_kernels,
                cnn_channels=cnn_channels,
                fusion_dim=fusion_dim,
                head_dropout=head_dropout,
                cnn_pooling=cnn_pooling,
                normalize_branches=normalize_branches,
                residual_gate=residual_gate,
                loss_type=loss_type,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
                num_labels=3,
                label2id=LABEL2ID,
                id2label=ID2LABEL,
            )
            model = cls(config)
            encoder, loading_info = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                token=token,
                add_pooling_layer=False,
                attn_implementation=attn_implementation,
                output_loading_info=True,
            )
            result = model.encoder.load_state_dict(encoder.state_dict(), strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(f"Encoder copy mismatch: {result}")
            del encoder
            return model, loading_info

        def _pooled(self, hidden_states, attention_mask):
            variant = self.config.variant
            mean = masked_mean(hidden_states, attention_mask)
            if variant == "mean":
                return mean
            local = self.local_projection(self.local_pool(hidden_states, attention_mask))
            if self.config.normalize_branches:
                local = self.local_norm(F.gelu(local))
            else:
                local = torch.tanh(local)
            if variant == "cnn":
                return local
            global_vector = self.global_projection(mean)
            if self.config.normalize_branches:
                global_vector = self.global_norm(F.gelu(global_vector))
            else:
                global_vector = torch.tanh(global_vector)
            joined = torch.cat([global_vector, local], dim=-1)
            if variant == "concat":
                return joined
            alpha = torch.sigmoid(self.gate(joined))
            fused = alpha * global_vector + (1.0 - alpha) * local
            if self.config.residual_gate:
                fused = self.fusion_norm(fused + 0.5 * (global_vector + local))
            return fused

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            **kwargs,
        ):
            if attention_mask is None:
                raise ValueError("attention_mask is required for controlled pooling")
            kwargs.pop("num_items_in_batch", None)
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )
            pooled = self._pooled(outputs.last_hidden_state, attention_mask)
            logits = self.classifier(self.dropout(pooled))
            loss = None
            if labels is not None:
                weights = (
                    self.class_weight_tensor.to(logits.dtype)
                    if self.class_weight_tensor.numel()
                    else None
                )
                if self.config.loss_type == "focal":
                    loss = focal_cross_entropy(
                        logits,
                        labels,
                        class_weights=weights,
                        gamma=self.config.focal_gamma,
                    )
                else:
                    loss = F.cross_entropy(logits, labels, weight=weights)
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

else:  # pragma: no cover - only used when dependencies are unavailable

    class GlobalLocalConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_model_dependencies()

    class MaskedCnnPool:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_model_dependencies()

    class ViSoBERTGlobalLocalForSequenceClassification:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_model_dependencies()


def build_model(
    config: Mapping,
    *,
    token: str | None = None,
    class_weights: Sequence[float] | None = None,
):
    """Build a fresh model from the locked pretrained encoder checkpoint."""
    require_model_dependencies()
    variant = config["variant"]
    common = {
        "revision": config["model_revision"],
        "token": token,
    }
    if variant == "standard":
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            config["model_id"],
            num_labels=3,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            attn_implementation=config.get("attention_implementation", "eager"),
            output_loading_info=True,
            **common,
        )
        return model, loading_info
    model, loading_info = ViSoBERTGlobalLocalForSequenceClassification.from_encoder_pretrained(
        config["model_id"],
        variant=variant,
        cnn_kernels=config.get("cnn_kernels", (2, 3, 4)),
        cnn_channels=int(config.get("cnn_channels", 64)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        head_dropout=float(config.get("head_dropout", 0.1)),
        cnn_pooling=str(config.get("cnn_pooling", "max")),
        normalize_branches=bool(config.get("normalize_branches", False)),
        residual_gate=bool(config.get("residual_gate", False)),
        loss_type=str(config.get("loss_type", "cross_entropy")),
        class_weights=class_weights,
        focal_gamma=float(config.get("focal_gamma", 0.0)),
        attn_implementation=config.get("attention_implementation", "eager"),
        **common,
    )
    return model, loading_info


def load_model_artifact(path: str | Path, *, token: str | None = None):
    require_model_dependencies()
    path = Path(path)
    raw = (path / "config.json").read_text(encoding="utf-8")
    import json
    config = json.loads(raw)
    if config.get("model_type") == GlobalLocalConfig.model_type:
        return ViSoBERTGlobalLocalForSequenceClassification.from_pretrained(
            path, local_files_only=True
        )
    return AutoModelForSequenceClassification.from_pretrained(
        path, token=token, local_files_only=True
    )


def count_parameters(model) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
