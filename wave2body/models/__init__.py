# -*- coding: utf-8 -*-
"""Model modules for mmWave sensor tokens and body-part pose tokens."""

from __future__ import annotations

__all__ = [
    "PointTransformer",
    "JointTokenizer",
    "PartJointTokenizer",
    "JointVQVAELoss",
    "CrossAttentionTokenTranslator",
    "TokenHPEEstimatorModel",
    "AbsoluteRootHeadEstimatorModel",
    "AbsoluteRootHead",
    "decode_part_token_logits",
    "PTPoseModel",
    "PoseMPJPELoss",
    "PointMaeTokenizer",
    "create_mmwave_tokenizer",
]


_EXPORTS = {
    "PointTransformer": ("point_transformer", "PointTransformer"),
    "JointTokenizer": ("human_body_tokenizer", "JointTokenizer"),
    "PartJointTokenizer": ("human_body_tokenizer", "PartJointTokenizer"),
    "JointVQVAELoss": ("human_body_tokenizer", "JointVQVAELoss"),
    "CrossAttentionTokenTranslator": ("token_translation", "CrossAttentionTokenTranslator"),
    "TokenHPEEstimatorModel": ("hpe_token", "TokenHPEEstimatorModel"),
    "AbsoluteRootHeadEstimatorModel": (
        "hpe_token",
        "AbsoluteRootHeadEstimatorModel",
    ),
    "AbsoluteRootHead": ("hpe_token", "AbsoluteRootHead"),
    "decode_part_token_logits": ("hpe_token", "decode_part_token_logits"),
    "PTPoseModel": ("point_transformer.pose_baseline", "PTPoseModel"),
    "PoseMPJPELoss": ("point_transformer.pose_baseline", "PoseMPJPELoss"),
    "PointMaeTokenizer": ("mmwave_tokenizer", "PointMaeTokenizer"),
    "create_mmwave_tokenizer": ("mmwave_tokenizer", "create_mmwave_tokenizer"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
