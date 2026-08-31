"""Build model-agnostic VR target-hand visualization payloads."""

from typing import Any, Optional

import numpy as np


# TeleVuer/WebXR hand landmark topology. The four non-thumb fingers contain
# an extra metacarpal landmark that some retargeting configs intentionally skip.
HAND_SKELETON_EDGES = np.asarray(
    [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
        (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
        (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
    ],
    dtype=np.int32,
)


def build_target_hand_payload(
    hand_points: np.ndarray,
    retargeting,
    *,
    side: str,
    anchor_body_name: Optional[str] = None,
) -> dict[str, Any]:
    """Describe landmarks, skeleton, and the optimizer's current constraints.

    ``hand_points`` must already use the coordinate convention expected by the
    selected hand retargeter. Points are centered at the VR wrist and uniformly
    scaled exactly as the optimizer scales reference vectors.
    """
    points = np.asarray(hand_points, dtype=np.float64)
    if points.shape != (25, 3):
        raise ValueError(f"Target-hand visualization expects (25, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Target-hand visualization received non-finite landmarks")
    if side not in {"left", "right"}:
        raise ValueError(f"Target-hand side must be left or right, got {side!r}")

    optimizer = retargeting.optimizer
    scaling = float(getattr(optimizer, "scaling", 1.0))
    local_landmarks = (points - points[0]) * scaling
    payload: dict[str, Any] = {
        "version": 1,
        "side": side,
        "optimizer_type": str(getattr(optimizer, "retargeting_type", "unknown")).lower(),
        "anchor_body_name": anchor_body_name,
        "landmarks": local_landmarks.tolist(),
        "skeleton_edges": HAND_SKELETON_EDGES.tolist(),
        "constraint_origin_indices": [],
        "constraint_task_indices": [],
        "constraint_vectors": [],
        "constraint_projected": [],
    }

    indices = np.asarray(
        getattr(optimizer, "target_link_human_indices", np.empty((2, 0), dtype=np.int32))
    )
    if indices.ndim != 2 or indices.shape[0] != 2:
        return payload

    origin_indices = indices[0].astype(np.int32)
    task_indices = indices[1].astype(np.int32)
    if np.any(origin_indices < 0) or np.any(origin_indices >= len(points)):
        raise ValueError("Constraint origin landmark index is outside [0, 24]")
    if np.any(task_indices < 0) or np.any(task_indices >= len(points)):
        raise ValueError("Constraint task landmark index is outside [0, 24]")

    raw_vectors = points[task_indices] - points[origin_indices]
    constraint_vectors = raw_vectors * scaling
    projected = np.zeros(len(raw_vectors), dtype=bool)

    # DexPilot updates this state while retargeting the current frame. Rebuild
    # its actual reference vectors so the visualization shows eta1/eta2 targets,
    # rather than only the original human fingertip distances.
    optimizer_projected = np.asarray(getattr(optimizer, "projected", []), dtype=bool)
    projected_dist = np.asarray(getattr(optimizer, "projected_dist", []), dtype=np.float64)
    if (
        optimizer_projected.ndim == 1
        and optimizer_projected.size > 0
        and optimizer_projected.size <= len(raw_vectors)
        and projected_dist.shape == optimizer_projected.shape
    ):
        projected[: len(optimizer_projected)] = optimizer_projected
        vector_norms = np.linalg.norm(raw_vectors[: len(projected_dist)], axis=1)
        directions = raw_vectors[: len(projected_dist)] / (vector_norms[:, None] + 1e-6)
        projected_vectors = directions * projected_dist[:, None]
        constraint_vectors[: len(projected_dist)] = np.where(
            optimizer_projected[:, None],
            projected_vectors,
            constraint_vectors[: len(projected_dist)],
        )

    payload.update(
        {
            "constraint_origin_indices": origin_indices.tolist(),
            "constraint_task_indices": task_indices.tolist(),
            "constraint_vectors": constraint_vectors.tolist(),
            "constraint_projected": projected.tolist(),
        }
    )
    return payload
