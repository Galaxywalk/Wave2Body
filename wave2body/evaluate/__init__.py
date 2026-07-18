"""Evaluation utilities for Wave2Body pose and token diagnostics."""

from importlib import import_module

__all__ = [
    "M4HumanActionGroupAccumulator",
    "M4HumanActionGroupVectorAccumulator",
    "m4human_action_ids_from_indicator",
    "summarize_vector_metric_by_action_group",
    "candidate_indices_and_nll",
    "decode_from_global_indices",
    "select_candidate_indices",
    "per_joint_error_mm",
    "per_sample_mpjpe_mm",
    "per_sample_p_mpjpe_mm",
    "PartTokenTopKAccumulator",
    "TokenAlignmentCollector",
    "build_different_action_same_subject_permutation",
    "build_same_action_cross_subject_permutation",
    "spearman_rank_correlation",
    "summarize_exact_agreement",
    "parse_topk_list",
    "mmwave_tokenizer_diagnostics",
    "evaluate_human_pose_tokenizer_checkpoint",
    "fit_joints_to_smpl_mesh",
    "JointToMeshFitResult",
    "JointToSMPLMeshFitter",
    "print_human_pose_tokenizer_reconstruction_summary",
    "save_human_pose_tokenizer_reconstruction",
    "save_meshes_as_obj",
]


_EXPORTS = {
    "M4HumanActionGroupAccumulator": ("m4human_action_groups", "M4HumanActionGroupAccumulator"),
    "M4HumanActionGroupVectorAccumulator": ("m4human_action_groups", "M4HumanActionGroupVectorAccumulator"),
    "m4human_action_ids_from_indicator": ("m4human_action_groups", "m4human_action_ids_from_indicator"),
    "summarize_vector_metric_by_action_group": ("m4human_action_groups", "summarize_vector_metric_by_action_group"),
    "candidate_indices_and_nll": ("posterior", "candidate_indices_and_nll"),
    "decode_from_global_indices": ("posterior", "decode_from_global_indices"),
    "select_candidate_indices": ("posterior", "select_candidate_indices"),
    "per_joint_error_mm": ("sample_metrics", "per_joint_error_mm"),
    "per_sample_mpjpe_mm": ("sample_metrics", "per_sample_mpjpe_mm"),
    "per_sample_p_mpjpe_mm": ("sample_metrics", "per_sample_p_mpjpe_mm"),
    "PartTokenTopKAccumulator": ("token_accuracy", "PartTokenTopKAccumulator"),
    "TokenAlignmentCollector": ("token_alignment", "TokenAlignmentCollector"),
    "build_different_action_same_subject_permutation": (
        "token_alignment",
        "build_different_action_same_subject_permutation",
    ),
    "build_same_action_cross_subject_permutation": (
        "token_alignment",
        "build_same_action_cross_subject_permutation",
    ),
    "spearman_rank_correlation": ("token_alignment", "spearman_rank_correlation"),
    "summarize_exact_agreement": ("token_alignment", "summarize_exact_agreement"),
    "parse_topk_list": ("token_accuracy", "parse_topk_list"),
    "evaluate_human_pose_tokenizer_checkpoint": ("human_pose_tokenizer_reconstruction", "evaluate_checkpoint"),
    "fit_joints_to_smpl_mesh": ("joint_to_mesh", "fit_joints_to_smpl_mesh"),
    "JointToMeshFitResult": ("joint_to_mesh", "JointToMeshFitResult"),
    "JointToSMPLMeshFitter": ("joint_to_mesh", "JointToSMPLMeshFitter"),
    "print_human_pose_tokenizer_reconstruction_summary": ("human_pose_tokenizer_reconstruction", "print_summary"),
    "save_human_pose_tokenizer_reconstruction": ("human_pose_tokenizer_reconstruction", "save_evaluation"),
    "save_meshes_as_obj": ("joint_to_mesh", "save_meshes_as_obj"),
}


def __getattr__(name: str):
    if name == "mmwave_tokenizer_diagnostics":
        return import_module(f"{__name__}.mmwave_tokenizer_diagnostics")
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
