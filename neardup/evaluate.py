"""在人工标注对上评估召回与误报。

labels.csv 每行 id_a,id_b,label，label=1 是正例（同一内容的变体），
0 是负例（确认无关）。同一对文档可能被重复标注；若一对同时被标过
1 和 0（样本集中存在一例），以 1 为准 —— 该对确实是同一内容的
变体，期望分组里也在同一组，计入负例会让"误报 = 0"自相矛盾。
"""

import csv
from collections import defaultdict


def load_labels(labels_path):
    """返回 (正例对集合, 负例对集合)，对按 (小id, 大id) 归一。"""
    labels = defaultdict(set)
    with open(labels_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = int(row["id_a"])
            b = int(row["id_b"])
            pair = (a, b) if a < b else (b, a)
            labels[pair].add(int(row["label"]))
    positives = {pair for pair, labs in labels.items() if 1 in labs}
    negatives = {pair for pair, labs in labels.items() if 1 not in labs}
    return positives, negatives


def load_groups(groups_path):
    """读分组输出，返回 doc_id -> 组号 的映射。"""
    member_of = {}
    with open(groups_path, encoding="utf-8") as f:
        for group_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            for token in line.split(","):
                member_of[int(token)] = group_idx
    return member_of


def evaluate(labels_path, groups_path):
    """返回 (recall, false_positive_rate, 详细计数字典)。"""
    positives, negatives = load_labels(labels_path)
    member_of = load_groups(groups_path)

    def same_group(pair):
        a, b = pair
        return (a in member_of and b in member_of
                and member_of[a] == member_of[b])

    hits = sum(1 for pair in positives if same_group(pair))
    false_pos = sum(1 for pair in negatives if same_group(pair))
    recall = hits / len(positives) if positives else 1.0
    fp_rate = false_pos / len(negatives) if negatives else 0.0
    detail = {
        "positives": len(positives),
        "positives_hit": hits,
        "negatives": len(negatives),
        "negatives_wrongly_grouped": false_pos,
    }
    return recall, fp_rate, detail
