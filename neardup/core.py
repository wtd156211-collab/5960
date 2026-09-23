"""归一化、特征提取与精确相似度。判定口径见 README。"""


def normalize_text(text):
    """去掉所有空白字符，ASCII 字母转小写，其余字符（含标点）原样保留。"""
    out = []
    for ch in text:
        if ch.isspace():
            continue
        if "A" <= ch <= "Z":
            ch = chr(ord(ch) + 32)
        out.append(ch)
    return "".join(out)


def extract_features(normalized):
    """字符级 3-gram 集合。长度 < 3 时整串为唯一特征；空串特征集为空。"""
    n = len(normalized)
    if n == 0:
        return frozenset()
    if n < 3:
        return frozenset((normalized,))
    return frozenset(normalized[i:i + 3] for i in range(n - 2))


def jaccard(feats_a, feats_b):
    """Jaccard 系数；两个空集之间定义为 1，空与非空之间为 0。"""
    if not feats_a:
        return 1.0 if not feats_b else 0.0
    if not feats_b:
        return 0.0
    inter = len(feats_a & feats_b)
    return inter / (len(feats_a) + len(feats_b) - inter)
