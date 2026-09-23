"""内容库近重复检测：MinHash/LSH 找候选 + 精确 Jaccard 复核 + 连通分量分组。"""

from .evaluate import evaluate
from .normalize import extract_features, jaccard, normalize
from .pipeline import Config, Stats, find_duplicate_groups

__all__ = [
    "Config",
    "Stats",
    "evaluate",
    "extract_features",
    "find_duplicate_groups",
    "jaccard",
    "normalize",
]

__version__ = "0.1.0"
