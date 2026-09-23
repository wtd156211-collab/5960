"""内容库近重复检测：前缀过滤候选 + 精确 Jaccard + 连通分量。"""

from .core import extract_features, jaccard, normalize_text
from .detect import Detector


def detect(docs_path, output_path=None, threshold=0.8, workdir=None):
    """便捷入口：对 docs_path 跑检测，返回分组列表。"""
    return Detector(threshold=threshold).detect(docs_path, output_path, workdir)


__all__ = ["Detector", "detect", "normalize_text", "extract_features", "jaccard"]
