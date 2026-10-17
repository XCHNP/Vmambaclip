"""VisionMambaCLIP components with explicit paper/engineering boundaries."""

from .adapters import TeacherAdapter, VimAdapter
from .checkpoint import (
    load_stage_checkpoint,
    read_checkpoint_metadata,
    required_teacher_stage,
    save_stage_checkpoint,
    validate_checkpoint_identity,
)
from .config import load_config, resolve_warmup_steps, validate_config
from .data import CANONICAL_EMOTION_ORDER, FERDataset, FERProtocol, build_protocol
from .losses import (
    Stage1Loss,
    Stage1LossConfig,
    Stage2Loss,
    Stage2LossConfig,
    duplicate_label_statistics,
    teacher_classification_loss,
    teacher_contrastive_loss,
)
from .model import HybridVisionMambaStudent, SASF, StudentOutput
from .metrics import ClassificationMetrics
from .stages import (
    configure_student_stage,
    configure_teacher_stage,
    freeze_teacher_for_student,
)
from .text_anchors import (
    PromptSchema,
    TextAnchorBank,
    class_name_ablation_schema,
    encode_prompt_schema,
    load_prompt_schema,
)
from .vim_backbone import BidirectionalVim, create_vim
from .official_vmamba import VMambaSpatialAdapter, create_official_vmamba

__all__ = [
    "TeacherAdapter",
    "VimAdapter",
    "HybridVisionMambaStudent",
    "SASF",
    "VMambaSpatialAdapter",
    "create_official_vmamba",
    "StudentOutput",
    "ClassificationMetrics",
    "Stage1Loss",
    "Stage1LossConfig",
    "Stage2Loss",
    "Stage2LossConfig",
    "FERDataset",
    "FERProtocol",
    "CANONICAL_EMOTION_ORDER",
    "build_protocol",
    "PromptSchema",
    "TextAnchorBank",
    "BidirectionalVim",
    "create_vim",
    "configure_student_stage",
    "configure_teacher_stage",
    "freeze_teacher_for_student",
    "teacher_contrastive_loss",
    "teacher_classification_loss",
    "duplicate_label_statistics",
    "load_prompt_schema",
    "encode_prompt_schema",
    "class_name_ablation_schema",
    "load_stage_checkpoint",
    "read_checkpoint_metadata",
    "save_stage_checkpoint",
    "validate_checkpoint_identity",
    "required_teacher_stage",
    "load_config",
    "validate_config",
    "resolve_warmup_steps",
]
