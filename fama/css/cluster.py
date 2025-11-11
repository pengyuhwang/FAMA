"""README 中描述的 CSS 聚类辅助函数。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_factors_kmeans(
    factor_matrix: "np.ndarray", k: int
) -> tuple[list[list[int]], "np.ndarray", "np.ndarray"]:
    """使用 KMeans 对因子暴露做聚类（README “CSS Context Assembly”）。

    Args:
        factor_matrix: README CSS 小节中提到的因子矩阵。
        k: 目标聚类数量，对应 defaults.yaml 中的 ``k``。

    Returns:
        (簇列表, 簇心, 标准化后的因子矩阵)。
    """

    if factor_matrix.size == 0:
        empty = np.zeros((0, 0))
        return [], empty, empty

    safe_matrix = _prepare_matrix(factor_matrix)
    n_obs = safe_matrix.shape[1]
    n_clusters = max(1, min(k, n_obs))
    if n_clusters == 1 or n_obs == 1:
        clusters = [list(range(n_obs))]
        centers = safe_matrix[:, :1].T if safe_matrix.size else np.zeros((1, safe_matrix.shape[0]))
        return clusters, centers, safe_matrix

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(safe_matrix.T)
    raw_clusters = [np.where(labels == idx)[0].tolist() for idx in range(n_clusters)]
    clusters = [cluster for cluster in raw_clusters if cluster]
    centers = []
    for members in clusters:
        cluster_matrix = safe_matrix[:, members]
        centers.append(cluster_matrix.mean(axis=1))
    centers_array = np.vstack(centers) if centers else np.zeros((0, safe_matrix.shape[0]))
    centers = centers_array
    return clusters, centers, safe_matrix


def select_cross_samples(
    clusters: list[list[int]],
    centers: "np.ndarray",
    factor_matrix: "np.ndarray",
    n_select: int,
) -> list[int]:
    """执行 SelectSamples 阶段，从每个簇挑选最靠近簇心的因子索引。

    Args:
        clusters: :func:`cluster_factors_kmeans` 的簇集合。
        centers: 对应簇心。
        factor_matrix: 标准化后的因子矩阵。
        n_select: 需要选择的因子数量（超参数 n）。

    Returns:
        传递给 CoE 的索引列表。
    """

    if not clusters or n_select <= 0:
        return []

    n_clusters = len(clusters)
    ordered_members: list[list[int]] = []
    for cluster_idx, members in enumerate(clusters):
        centroid = centers[cluster_idx]
        ranked = sorted(
            members,
            key=lambda member: np.linalg.norm(factor_matrix[:, member] - centroid),
        )
        ordered_members.append(ranked)

    selections: list[int] = []
    exhausted = [False] * n_clusters
    cursor = 0
    while len(selections) < n_select and not all(exhausted):
        cluster_idx = cursor % n_clusters
        if not ordered_members[cluster_idx]:
            exhausted[cluster_idx] = True
            cursor += 1
            continue
        candidate = ordered_members[cluster_idx].pop(0)
        if not ordered_members[cluster_idx]:
            exhausted[cluster_idx] = True
        if candidate not in selections:
            selections.append(candidate)
        cursor += 1

    return selections


def _prepare_matrix(matrix: "np.ndarray") -> "np.ndarray":
    """对因子矩阵进行裁剪与标准化，避免 KMeans 出现溢出。"""

    safe = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    safe = np.clip(safe, -1e6, 1e6)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(safe)
    return np.nan_to_num(scaled, nan=0.0)
