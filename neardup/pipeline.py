"""近重复检测主流程。

五个阶段，每个阶段都是流式 / 分块的，任何一步都不把全部文档的
完整特征或全部文档对堆进内存：

1. 顺序扫描输入：归一化 -> 特征 -> MinHash 签名。归一化文本写入
   临时文件（内存只留偏移量和特征数），同时流出两类定长记录：
   全文指纹记录（折叠完全重复）和分带记录（LSH 候选）。
2. 外排序指纹记录，相同指纹（归一化文本完全一致，含空文档）的
   文档直接并查集合并 —— 它们的特征集相同，相似度恒为 1。
3. 外排序分带记录，按带键分桶流式产出候选对（桶内先映射到
   完全重复组的代表，去掉冗余），候选对写入临时文件。
4. 外排序候选对并去重，逐对流式复核：先用特征数上下界剪枝，
   再从磁盘只读出这两篇的文本、重算特征集、算精确 Jaccard，
   >= 阈值才合并。近似手段只用于找候选，判定永远是精确的。
5. 并查集连通分量 -> 按规则输出分组。
"""

import json
import os
import struct
import tempfile
from array import array
from dataclasses import dataclass
from functools import lru_cache

from .externalsort import sort_records
from .minhash import BANDS, band_keys, signature, text_fingerprint
from .normalize import extract_features, jaccard, normalize
from .store import TextStore
from .unionfind import UnionFind

_ORD = struct.Struct(">I")    # 文档序号，大端保证字节序 == 数值序
_PAIR = struct.Struct(">II")
_HASH_RECORD = 16 + _ORD.size          # 全文指纹记录：16 字节指纹 + 序号
_BAND_RECORD = 16 + _ORD.size          # 分带记录：16 字节带键 + 序号
_PAIR_RECORD = _PAIR.size


@dataclass
class Config:
    threshold: float = 0.8        # 判定阈值 τ
    chunk_records: int = 1_000_000  # 外排序每个分块的记录数（内存上限的旋钮）
    max_bucket: int = 1_000_000   # 单个 LSH 桶去重后代表数的安全上限
    cache_size: int = 1024        # 复核阶段的特征集 LRU 缓存容量
    temp_dir: str = None          # 临时文件目录，None 用系统默认


@dataclass
class Stats:
    n_docs: int = 0
    n_exact_dup_unions: int = 0
    n_candidate_pairs: int = 0
    n_verified_pairs: int = 0
    n_merged_pairs: int = 0
    n_groups: int = 0


def find_duplicate_groups(input_path, output_path, config=None):
    """跑完整流程，把分组写进 output_path，返回 Stats。"""
    config = config or Config()
    tempdir = tempfile.mkdtemp(prefix="neardup_", dir=config.temp_dir)
    stats = Stats()
    try:
        paths = {name: os.path.join(tempdir, name)
                 for name in ("texts.bin", "hash.raw", "hash.sorted",
                              "band.raw", "band.sorted",
                              "pairs.raw", "pairs.sorted")}
        ids = array("q")
        store = TextStore(paths["texts.bin"])
        _pass1_scan(input_path, store, ids, paths)
        stats.n_docs = len(ids)
        uf = UnionFind(len(ids))
        _pass2_exact_duplicates(paths, uf, config, stats)
        _pass3_candidates(paths, uf, config, stats)
        _pass4_verify(paths, store, uf, config, stats)
        store.close()
        _pass5_emit(ids, uf, output_path, stats)
        return stats
    finally:
        for name in os.listdir(tempdir):
            os.unlink(os.path.join(tempdir, name))
        os.rmdir(tempdir)


def _pass1_scan(input_path, store, ids, paths):
    ordinal = 0
    with open(input_path, "r", encoding="utf-8") as src, \
            open(paths["hash.raw"], "wb") as hash_out, \
            open(paths["band.raw"], "wb") as band_out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            ids.append(doc["id"])
            normalized = normalize(doc.get("text") or "")
            features = extract_features(normalized)
            store.append(normalized, len(features))
            hash_out.write(text_fingerprint(normalized))
            hash_out.write(_ORD.pack(ordinal))
            if features:  # 空文档没有特征，靠指纹记录两两成组
                for key in band_keys(signature(features)):
                    band_out.write(key)
                    band_out.write(_ORD.pack(ordinal))
            ordinal += 1
    store.finish_writing()


def _pass2_exact_duplicates(paths, uf, config, stats):
    sort_records(paths["hash.raw"], paths["hash.sorted"], _HASH_RECORD,
                 tempdir=os.path.dirname(paths["hash.raw"]),
                 chunk_records=config.chunk_records)
    group = []
    with open(paths["hash.sorted"], "rb") as f:
        prev_key = None
        while True:
            record = f.read(_HASH_RECORD)
            if not record:
                break
            key = record[:16]
            ordinal = _ORD.unpack(record[16:])[0]
            if key != prev_key:
                stats.n_exact_dup_unions += _union_group(uf, group)
                group = []
                prev_key = key
            group.append(ordinal)
        stats.n_exact_dup_unions += _union_group(uf, group)


def _union_group(uf, ordinals):
    merges = 0
    for other in ordinals[1:]:
        if uf.find(ordinals[0]) != uf.find(other):
            uf.union(ordinals[0], other)
            merges += 1
    return merges


def _pass3_candidates(paths, uf, config, stats):
    tempdir = os.path.dirname(paths["band.raw"])
    sort_records(paths["band.raw"], paths["band.sorted"], _BAND_RECORD,
                 tempdir=tempdir, chunk_records=config.chunk_records)
    bucket = []
    with open(paths["band.sorted"], "rb") as src, \
            open(paths["pairs.raw"], "wb") as out:
        prev_key = None
        while True:
            record = src.read(_BAND_RECORD)
            if not record:
                break
            key = record[:16]
            ordinal = _ORD.unpack(record[16:])[0]
            if key != prev_key:
                stats.n_candidate_pairs += _emit_bucket(uf, bucket, out,
                                                        config)
                bucket = []
                prev_key = key
            bucket.append(ordinal)
        stats.n_candidate_pairs += _emit_bucket(uf, bucket, out, config)


def _emit_bucket(uf, bucket, out, config):
    """桶内序号升序到达；映射到完全重复组的代表后两两配对。"""
    if len(bucket) < 2:
        return 0
    reps = []
    seen = set()
    for ordinal in bucket:
        rep = uf.find(ordinal)
        if rep not in seen:
            seen.add(rep)
            reps.append(rep)
    n = len(reps)
    if n < 2:
        return 0
    if n > config.max_bucket:
        raise RuntimeError(
            "LSH 桶过大（%d 个代表，上限 %d）：请增大每带行数或调参"
            % (n, config.max_bucket))
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            out.write(_PAIR.pack(reps[i], reps[j]))
            pairs += 1
    return pairs


def _pass4_verify(paths, store, uf, config, stats):
    tempdir = os.path.dirname(paths["pairs.raw"])
    sort_records(paths["pairs.raw"], paths["pairs.sorted"], _PAIR_RECORD,
                 tempdir=tempdir, chunk_records=config.chunk_records,
                 dedup=True)
    threshold = config.threshold

    @lru_cache(maxsize=config.cache_size)
    def features_of(ordinal):
        return extract_features(store.text(ordinal))

    counts = store.counts
    with open(paths["pairs.sorted"], "rb") as f:
        while True:
            record = f.read(_PAIR_RECORD)
            if not record:
                break
            a, b = _PAIR.unpack(record)
            ca = counts[a]
            cb = counts[b]
            # J <= min(|A|,|B|)/max(|A|,|B|)，不够界的直接跳过
            if min(ca, cb) < threshold * max(ca, cb):
                continue
            stats.n_verified_pairs += 1
            if jaccard(features_of(a), features_of(b)) >= threshold:
                uf.union(a, b)
                stats.n_merged_pairs += 1


def _pass5_emit(ids, uf, output_path, stats):
    groups = {}
    for ordinal in range(len(ids)):
        root = uf.find(ordinal)
        groups.setdefault(root, []).append(ordinal)
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        rows.append(sorted(ids[o] for o in members))
    rows.sort(key=lambda member_ids: member_ids[0])
    stats.n_groups = len(rows)
    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        for member_ids in rows:
            out.write(",".join(str(doc_id) for doc_id in member_ids))
            out.write("\n")
