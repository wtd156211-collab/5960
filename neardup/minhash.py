"""MinHash 指纹与 LSH 分带。

只用于产生候选对；最终判定永远回到精确 Jaccard（见 pipeline）。
所有常量由固定种子生成，跨进程、跨平台确定。
"""

import hashlib
import random
import struct

BANDS = 32          # 分带数 b
ROWS = 6            # 每带行数 r
PERMUTATIONS = BANDS * ROWS  # 签名长度 192

_MASK32 = (1 << 32) - 1

_rng = random.Random(0x9E3779B9)
_A = tuple(_rng.getrandbits(32) | 1 for _ in range(PERMUTATIONS))
_B = tuple(_rng.getrandbits(32) for _ in range(PERMUTATIONS))

_BAND_STRUCT = struct.Struct("<%dI" % ROWS)


def _feature_hash(feature):
    """单个 3-gram 的 64 位基哈希（blake2b，确定性强）。"""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "little")


def signature(features):
    """非空特征集的 MinHash 签名，返回长度为 PERMUTATIONS 的列表。"""
    sig = [_MASK32] * PERMUTATIONS
    perm_a = _A
    perm_b = _B
    mask = _MASK32
    n_perm = PERMUTATIONS
    for feature in features:
        h = _feature_hash(feature)
        for i in range(n_perm):
            v = (perm_a[i] * h + perm_b[i]) & mask
            if v < sig[i]:
                sig[i] = v
    return sig


def band_keys(sig):
    """把签名切成 BANDS 个带，每带算一个 16 字节的确定性强哈希键。"""
    keys = []
    for band in range(BANDS):
        chunk = _BAND_STRUCT.pack(*sig[band * ROWS:(band + 1) * ROWS])
        keys.append(hashlib.blake2b(chunk, digest_size=16,
                                    person=bytes([band])).digest())
    return keys


def text_fingerprint(normalized):
    """归一化全文的 128 位指纹，用于折叠完全重复（含空文档）。"""
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()
