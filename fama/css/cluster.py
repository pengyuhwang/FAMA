"""README 中描述的 CSS 聚类辅助函数。"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_factors_kmeans(factor_matrix: "np.ndarray", k: int) -> list[list[int]]:
    """使用 KMeans 对因子暴露做聚类（README “CSS Context Assembly”）。

    Args:
        factor_matrix: README CSS 小节中提到的因子矩阵。
        k: 目标聚类数量，对应 defaults.yaml 中的 ``k``。

    Returns:
        聚类结果，每个元素为索引列表。
    """

    if factor_matrix.size == 0:
        return []

    safe_matrix = _prepare_matrix(factor_matrix)
    n_obs = safe_matrix.shape[1]
    n_clusters = max(1, min(k, n_obs))
    if n_clusters == 1 or n_obs == 1:
        return [list(range(n_obs))]

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(safe_matrix.T)
    clusters = [np.where(labels == idx)[0].tolist() for idx in range(n_clusters)]
    return [cluster for cluster in clusters if cluster]


def select_cross_samples(clusters: list[list[int]], l: int) -> list[int]:
    """执行 SelectSamples 阶段，挑选多样化的因子索引。

    Args:
        clusters: :func:`cluster_factors_kmeans` 的输出。
        l: defaults.yaml 中的 ``l`` 值。

    Returns:
        传递给 CoE 的索引列表。
    """

    if not clusters:
        return []

    desired = max(1, l)
    selections: list[int] = []
    cluster_iter = 0
    while len(selections) < desired:
        cluster = clusters[cluster_iter % len(clusters)]
        pick = cluster[len(selections) % len(cluster)]
        selections.append(pick)
        cluster_iter += 1
        if len(selections) >= desired:
            break
    unique: list[int] = []
    seen: set[int] = set()
    for idx in selections:
        if idx in seen:
            continue
        seen.add(idx)
        unique.append(idx)
    return unique


def _prepare_matrix(matrix: "np.ndarray") -> "np.ndarray":
    """对因子矩阵进行裁剪与标准化，避免 KMeans 出现溢出。"""

    safe = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    safe = np.clip(safe, -1e6, 1e6)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(safe)
    return np.nan_to_num(scaled, nan=0.0)
