"""RAF-DB preprocessing: flat-dir + labels.txt → split CSVs + AU prompt schema.

This is the single source of truth for RAF-DB's label mapping, emotion names,
and AU feature descriptions.  Two consumers share these constants:

1. ``generate_split_csv()`` emits ``train_*/val_*/test_*.csv`` for the generic
   OpenCLIP training path (image_path / caption / labels).
2. ``write_prompt_schema()`` emits ``configs/rafdb_au_prompts.json`` for the
   VisionMambaCLIP runner, whose ``open_clip.vision_mamba_clip.data`` module
   reads the *same* flat-dir + labels.txt layout directly (see
   ``RAW_LABEL_TO_INDEX`` / ``CANONICAL_EMOTION_ORDER`` there) and re-derives
   the identical 90/10 train/val + full-test split.

学术黄金法则: every ``test_*`` image goes to the test set; ``train_*`` images are
split per class 90/10 into train/val.  No test image ever leaks into train/val.
"""

import os
import csv
import json
import random
from tqdm import tqdm

# ===== 配置路径 =====
data_name = "rafdb"
res_name = "0324"
DATASET_ROOT = "/root/autodl-tmp/open_clip-main/rafdb"
LABEL_TXT_PATH = "/root/autodl-tmp/open_clip-main/rafdb_labels.txt"
PROMPT_SCHEMA_PATH = "/root/autodl-tmp/open_clip-main/configs/rafdb_au_prompts.json"
TRAIN_CSV = f"/root/autodl-tmp/open_clip-main/src/open_clip_train/{data_name}/train_{res_name}.csv"
VAL_CSV   = f"/root/autodl-tmp/open_clip-main/src/open_clip_train/{data_name}/val_{res_name}.csv"
TEST_CSV  = f"/root/autodl-tmp/open_clip-main/src/open_clip_train/{data_name}/test_{res_name}.csv"

IMAGE_EXTS  = [".jpg", ".jpeg", ".png"]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ===== 标签映射（保持与 data.py 的 RAW_LABEL_TO_INDEX 一致） =====
# 原始标签 1..7 → 规范索引 0..6（RAF-DB 官方映射）
LABEL_MAPPING = {
    "1": "6", "2": "2", "3": "1", "4": "3", "5": "5", "6": "0", "7": "4"
}
# 规范索引 → 情感名（索引 0..6 固定，与 CANONICAL_EMOTION_ORDER 一致）
EMOTION_NAMES = {
    "0": "Angry", "1": "Disgusted", "2": "Fearful",
    "3": "Happy",  "4": "Neutral",   "5": "Sad", "6": "Surprised"
}

AU_FEATURES = {
    "0": {
        "upper": [
            "eyebrows are lowered and pulled together deeply",
            "brows are furrowed with vertical wrinkles between them",
            "inner eyebrows are drawn down and together in a frown",
            "intense gaze with lowered eyebrows and tightened eyelids"
        ],
        "middle": [
            "nostrils may be flared slightly",
            "eyes are glaring with tension in the lower eyelids",
            "vertical lines appear between the brows (glabella lines)"
        ],
        "lower": [
            "lips are pressed firmly together forming a straight line",
            "mouth is tightly closed with corners tense",
            "jaw is clenched and pushed forward",
            "lips look thin and compressed due to tension"
        ]
    },
    "1": {
        "upper": [
            "eyebrows are slightly lowered",
            "brows are drawn down, creating a crinkled forehead",
            "eyes are narrowed or squinting"
        ],
        "middle": [
            "nose is wrinkled significantly at the bridge",
            "deep nasolabial folds visible around the nose",
            "cheeks are raised, pushing up the lower eyelids"
        ],
        "lower": [
            "upper lip is raised, exposing the upper teeth",
            "mouth is open with the upper lip pulled up",
            "corners of the mouth are pulled down slightly",
            "lower lip is raised and protruding"
        ]
    },
    "2": {
        "upper": [
            "eyebrows are raised and pulled together straight",
            "inner brows are lifted high and drawn together",
            "forehead shows short horizontal wrinkles in the center"
        ],
        "middle": [
            "eyes are wide open with upper white sclera visible",
            "upper eyelids are raised high due to tension",
            "lower eyelids are tense and raised"
        ],
        "lower": [
            "lips are parted and stretched horizontally",
            "mouth corners are pulled back tightly",
            "jaw is dropped slightly with lips tense",
            "mouth is open in a tense, rectangular shape"
        ]
    },
    "3": {
        "upper": [
            "eyebrows are relaxed or slightly raised",
            "outer corners of the eyes exhibit crow's feet wrinkles",
            "skin around the eyes is crinkled"
        ],
        "middle": [
            "cheeks are raised high and prominent",
            "lower eyelids are pushed up by the cheeks",
            "distinct nasolabial folds run from nose to mouth"
        ],
        "lower": [
            "corners of the mouth are pulled upwards and backwards",
            "lips are parted exposing the upper teeth in a smile",
            "mouth forms a U-shape curve",
            "a wide smile with visible teeth"
        ]
    },
    "4": {
        "upper": [
            "eyebrows are in a resting, neutral position",
            "forehead is smooth with no wrinkles",
            "brows are relaxed without tension"
        ],
        "middle": [
            "eyes are open normally with a calm gaze",
            "eyelids are relaxed without drooping or widening",
            "cheeks and nose are relaxed"
        ],
        "lower": [
            "lips are touching lightly and relaxed",
            "mouth corners are neutral, neither up nor down",
            "jaw is relaxed and closed",
            "face shows no distinct muscle contraction"
        ]
    },
    "5": {
        "upper": [
            "inner corners of the eyebrows are raised and drawn together",
            "brows form an inverted 'V' shape",
            "forehead shows an omega-shaped wrinkle in the center"
        ],
        "middle": [
            "upper eyelids are drooping slightly",
            "eyes look downcast or watery",
            "cheeks are sagging or flat"
        ],
        "lower": [
            "corners of the mouth are pulled firmly downwards",
            "lips appear to be trembling or pouting",
            "chin boss is raised and wrinkled",
            "mouth forms an inverted U-shape"
        ]
    },
    "6": {
        "upper": [
            "eyebrows are raised high and curved",
            "long horizontal wrinkles cover the entire forehead",
            "brows are arched significantly"
        ],
        "middle": [
            "eyes are wide open with whites visible above and below iris",
            "upper eyelids are raised as high as possible",
            "no tension in the lower eyelids"
        ],
        "lower": [
            "jaw is dropped open, relaxing the mouth",
            "mouth forms a relaxed 'O' shape",
            "lips are parted but not stretched or tense",
            "mouth hangs open naturally"
        ]
    }
}


# ===== Prompt 生成 =====

def generate_training_prompt(label):
    f = AU_FEATURES[label]
    return (
        f"A photo of a face. "
        f"The {random.choice(f['upper'])}. "
        f"The {random.choice(f['middle'])}. "
        f"The {random.choice(f['lower'])}. "
        f"This is a {EMOTION_NAMES[label]} expression."
    )

def generate_inference_prompt(label):
    f = AU_FEATURES[label]
    return (
        f"A photo of a face. "
        f"The {f['upper'][0]}. "
        f"The {f['middle'][0]}. "
        f"The {f['lower'][0]}. "
        f"This is a {EMOTION_NAMES[label]} expression."
    )


def write_prompt_schema(output_path=PROMPT_SCHEMA_PATH):
    """Emit configs/rafdb_au_prompts.json from AU_FEATURES (single source of truth).

    The VisionMambaCLIP runner loads this file and validates its ``class_order``
    against ``CANONICAL_EMOTION_ORDER`` in ``open_clip.vision_mamba_clip.data``.
    Generating it here (same AU_FEATURES + template as the CSV captions) keeps the
    text anchors used by the runner identical to the training captions.
    """
    class_order = [EMOTION_NAMES[key] for key in sorted(EMOTION_NAMES, key=int)]
    prompts = {EMOTION_NAMES[key]: generate_inference_prompt(key) for key in EMOTION_NAMES}
    payload = {
        "class_order": class_order,
        "prompts": prompts,
        "source": (
            "RAF-DB AU prompts generated from predata_rafdb.py AU_FEATURES "
            "(upper/middle/lower first entries); canonical order 0-6 = "
            "Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised"
        ),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ AU prompt schema written: {output_path}")


# ===== 数据泄漏检查 =====

def check_data_leakage(train_data, val_data, test_data):
    train_paths = {p for p, _ in train_data}
    val_paths   = {p for p, _ in val_data}
    test_paths  = {p for p, _ in test_data}
    overlaps = (
        len(train_paths & val_paths) +
        len(train_paths & test_paths) +
        len(val_paths   & test_paths)
    )
    if overlaps == 0:
        print("✅ 数据集划分检查通过：无数据泄露")
    else:
        print(f"❌ 警告：发现数据泄露！重叠样本总数: {overlaps}")


# ===== 主处理逻辑 =====

def generate_split_csv(config_path=None, output_dir=None):
    """Use the same FERProtocol as the runner; refuse existing export directories."""
    from pathlib import Path
    from open_clip_train.predata_fer import preprocess

    project = Path(__file__).resolve().parents[2]
    return preprocess(
        str(config_path or project / "configs/rafdb_config.json"),
        str(output_dir or project / "artifacts/preprocessed/rafdb"),
    )


if __name__ == "__main__":
    from pathlib import Path
    from open_clip_train.predata_fer import main

    main(default_config=str(Path(__file__).resolve().parents[2] / "configs/rafdb_config.json"))
