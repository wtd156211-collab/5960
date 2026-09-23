"""归一化、特征提取与相似度计算（判定口径的精确实现）。"""

# Python str.isspace() 为真的全部码位（Unicode 15 之前的稳定集合）。
_WHITESPACE = (
    "\t\n\v\f\r "           # \x09-\x0d \x20
    "\x1c\x1d\x1e\x1f"      # 文件/组/记录/单元分隔符
    "\x85\xa0"              # NEL, NBSP
    "\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


def _build_table():
    table = {ord(ch): None for ch in _WHITESPACE}
    for cp in range(ord("A"), ord("Z") + 1):
        table[cp] = chr(cp + 32)  # 仅 ASCII 字母转小写，其余字符原样保留
    return str.maketrans(table)


_TABLE = _build_table()


def normalize(text):
    """去掉所有空白字符，ASCII 字母转小写，其余字符（含标点）原样保留。"""
    return text.translate(_TABLE)


def extract_features(normalized):
    """归一化文本的字符级 3-gram 集合。

    长度小于 3 时整串是唯一特征；长度为 0 时特征集为空集。
    """
    n = len(normalized)
    if n == 0:
        return frozenset()
    if n < 3:
        return frozenset((normalized,))
    return frozenset(normalized[i:i + 3] for i in range(n - 2))


def jaccard(feats_a, feats_b):
    """Jaccard 系数；两个空集之间定义为 1，空集与非空集之间为 0。"""
    if not feats_a:
        return 1.0 if not feats_b else 0.0
    if not feats_b:
        return 0.0
    inter = len(feats_a & feats_b)
    return inter / (len(feats_a) + len(feats_b) - inter)
