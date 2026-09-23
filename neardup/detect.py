"""近重复检测主管线：前缀过滤找候选 + 精确 Jaccard 复核 + 连通分量分组。

内存口径（详见 README）：
- 文档只流式读一遍，完整特征集落到临时特征文件，内存里只留
  (id, 偏移量) 两个紧凑数组；
- 候选用「按文档频率排序的前缀过滤」确定性产生（数学上不漏），
  全部中间结果（token 倒排、候选对）都通过分块外排序在磁盘上归并，
  内存里始终只有一个排序块；
- 复核时按偏移量从特征文件随用随取，配一个有界 LRU 缓存，
  任意时刻只有常数篇文档的特征在内存里。
"""

import json
import math
import os
import tempfile
from array import array
from collections import OrderedDict

from .core import extract_features, jaccard, normalize_text
from .externalsort import external_sort
from .unionfind import UnionFind

# 候选按 threshold - _TAU_EPS 建索引（前缀只会更长、候选只会更多），
# 吸收阈值附近的浮点误差；最终判定仍用精确的 threshold。
_TAU_EPS = 1e-9


def prefix_length(n, tau):
    """大小为 n 的特征集需要索引的前缀长度。

    若 Jaccard(A, B) >= tau，则 |A∩B| >= ceil(tau * max(|A|,|B|))。
    两边各自把特征按同一个全局序（这里用 token 的文档频率，频次相同
    按字典序）排序，各取前 n - ceil(tau*n) + 1 个作为前缀，则两边前缀
    必然共享至少一个特征（容斥可证）。因此按前缀 token 分桶不会漏检，
    与具体排序方式无关；按文档频率排序只是让候选更少。
    """
    p = n - math.ceil(tau * n) + 1
    return max(1, min(p, n))


class Detector:
    def __init__(self, threshold=0.8, sort_chunk_lines=200_000, cache_size=4096):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold 必须在 (0, 1] 之间")
        self.threshold = threshold
        self.sort_chunk_lines = sort_chunk_lines
        self.cache_size = cache_size

    def detect(self, docs_path, output_path=None, workdir=None):
        """运行检测，返回分组（list[list[int]]，组员与组序均按 README 规则）。

        output_path 不为 None 时同时把结果写入该文件（行尾 LF）。
        """
        if workdir is None:
            with tempfile.TemporaryDirectory(prefix="neardup_") as tmp:
                return self.detect(docs_path, output_path, tmp)

        paths = {name: os.path.join(workdir, name)
                 for name in ("features.bin", "alltokens.txt", "counts.txt",
                              "tokens.sorted", "bydoc.txt", "bydoc.sorted",
                              "counts.sorted", "prefix.txt", "prefix.sorted",
                              "pairs.raw", "pairs.txt")}

        ids, offsets, counts, empty_ids = self._pass1_scan(
            docs_path, paths["features.bin"], paths["alltokens.txt"], paths["counts.txt"])
        self._pass2_candidates(paths)
        uf = self._pass3_verify(paths["features.bin"], ids, offsets,
                                counts, paths["pairs.txt"], empty_ids)

        groups = self._format_groups(uf)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8", newline="") as fout:
                for g in groups:
                    fout.write(",".join(map(str, g)) + "\n")
        return groups

    # ---- 第一遍：流式扫描，特征与 token 记录落盘，内存只留 (id, offset) 数组 ----

    def _pass1_scan(self, docs_path, features_path, alltokens_path, counts_path):
        ids = array("q")
        offsets = array("q")
        counts = array("q")
        empty_ids = []
        with open(docs_path, "r", encoding="utf-8") as fin, \
                open(features_path, "wb") as ffeat, \
                open(alltokens_path, "w", encoding="utf-8", newline="") as ftok, \
                open(counts_path, "w", encoding="utf-8", newline="") as fcnt:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                doc_id = obj["id"]
                feats = extract_features(normalize_text(obj["text"]))
                offsets.append(ffeat.tell())
                ids.append(doc_id)
                counts.append(len(feats))
                tokens = sorted(feats)
                ffeat.write((" ".join(tokens) + "\n").encode("utf-8"))
                if not tokens:
                    empty_ids.append(doc_id)
                    continue
                fcnt.write("%020d\t%d\n" % (doc_id, len(tokens)))
                for tok in tokens:
                    ftok.write(tok + "\t" + str(doc_id) + "\n")
        return ids, offsets, counts, empty_ids

    # ---- 第二遍：外排序归并出候选对（全程磁盘，内存只有一个排序块）----

    def _pass2_candidates(self, paths):
        workdir = os.path.dirname(paths["alltokens.txt"])
        chunk = self.sort_chunk_lines
        tau_index = max(0.0, self.threshold - _TAU_EPS)

        # 1) 按 token 归并倒排，得到每个 token 的文档频率 df。
        external_sort(paths["alltokens.txt"], paths["tokens.sorted"], workdir, chunk_lines=chunk)
        with open(paths["tokens.sorted"], "r", encoding="utf-8", newline="") as fin, \
                open(paths["bydoc.txt"], "w", encoding="utf-8", newline="") as fout:
            for token, docs in self._groups_by_token(fin):
                df = len(docs)
                for doc_id in docs:
                    fout.write("%020d\t%010d\t%s\n" % (doc_id, df, token))

        # 2) 按 (doc, df, token) 归并，每篇文档的 token 按稀有度排好序。
        external_sort(paths["bydoc.txt"], paths["bydoc.sorted"], workdir, chunk_lines=chunk)
        external_sort(paths["counts.txt"], paths["counts.sorted"], workdir, chunk_lines=chunk)

        # 3) 每篇文档只保留最稀有的 prefix_length 个 token 作为前缀记录。
        with open(paths["bydoc.sorted"], "r", encoding="utf-8", newline="") as fdoc, \
                open(paths["counts.sorted"], "r", encoding="utf-8", newline="") as fcnt, \
                open(paths["prefix.txt"], "w", encoding="utf-8", newline="") as fout:
            counts_iter = iter(fcnt)
            count_line = next(counts_iter, None)
            for doc_key, lines in self._groups_by_doc(fdoc):
                while count_line is not None and count_line[:20] < doc_key:
                    count_line = next(counts_iter, None)
                if count_line is None or count_line[:20] != doc_key:
                    raise ValueError("counts 与 bydoc 不一致，doc=%s" % doc_key)
                keep = prefix_length(int(count_line[21:]), tau_index)
                for line in lines[:keep]:
                    fout.write(line[32:-1] + "\t" + str(int(doc_key)) + "\n")

        # 4) 前缀记录按 token 归并，同桶两两成对，去重后就是候选对。
        external_sort(paths["prefix.txt"], paths["prefix.sorted"], workdir, chunk_lines=chunk)
        with open(paths["prefix.sorted"], "r", encoding="utf-8", newline="") as fin, \
                open(paths["pairs.raw"], "w", encoding="utf-8", newline="") as fout:
            for _token, docs in self._groups_by_token(fin):
                ordered = sorted(docs)
                for i in range(len(ordered)):
                    for j in range(i + 1, len(ordered)):
                        # 零填充让字符串排序等价于数值排序：复核时同一篇文档的
                        # 候选对聚在一起，它的特征集只需从磁盘读一次。
                        fout.write("%020d\t%020d\n" % (ordered[i], ordered[j]))
        external_sort(paths["pairs.raw"], paths["pairs.txt"], workdir,
                      chunk_lines=chunk, unique=True)

    @staticmethod
    def _groups_by_token(lines):
        """把 'token\\tdoc' 有序行按 token 分组，产出 (token, [doc_id])。

        单个 token 的倒排表在内存里；前缀过滤保证高频 token 进不了前缀，
        所以这里只可能是该 token 真的稀有，表不会失控。
        """
        token, docs = None, []
        for line in lines:
            tok, _, doc = line.rpartition("\t")
            if tok != token and docs:
                yield token, docs
                docs = []
            token = tok
            docs.append(int(doc))
        if docs:
            yield token, docs

    @staticmethod
    def _groups_by_doc(lines):
        """把 '%020d\\t%010d\\ttoken' 有序行按 doc 分组，产出 (doc_key, [lines])。"""
        key, group = None, []
        for line in lines:
            doc_key = line[:20]
            if doc_key != key and group:
                yield key, group
                group = []
            key = doc_key
            group.append(line)
        if group:
            yield key, group

    # ---- 第三遍：精确复核，只把候选涉及的特征读进内存 ----

    def _pass3_verify(self, features_path, ids, offsets, counts, pairs_path, empty_ids):
        needed = set()
        with open(pairs_path, "r", encoding="utf-8") as fin:
            for line in fin:
                a, b = line.split("\t")
                needed.add(int(a))
                needed.add(int(b))
        offset_of = {}
        count_of = {}
        for idx in range(len(ids)):
            doc_id = ids[idx]
            if doc_id in needed:
                offset_of[doc_id] = offsets[idx]
                count_of[doc_id] = counts[idx]

        cache = OrderedDict()
        cache_size = self.cache_size
        threshold = self.threshold

        def features_of(doc_id):
            feats = cache.get(doc_id)
            if feats is not None:
                cache.move_to_end(doc_id)
                return feats
            ffeat.seek(offset_of[doc_id])
            line = ffeat.readline().decode("utf-8").rstrip("\n")
            feats = frozenset(line.split(" ")) if line else frozenset()
            cache[doc_id] = feats
            if len(cache) > cache_size:
                cache.popitem(last=False)
            return feats

        uf = UnionFind()
        tau_index = max(0.0, self.threshold - _TAU_EPS)
        with open(features_path, "rb") as ffeat, \
                open(pairs_path, "r", encoding="utf-8") as fin:
            for line in fin:
                a, b = line.split("\t")
                a, b = int(a), int(b)
                # 尺寸预过滤：J >= τ 要求 min(|A|,|B|) >= τ·max(|A|,|B|)，
                # 用特征数即可零成本排除，不必读特征。
                ca, cb = count_of[a], count_of[b]
                if ca > cb:
                    ca, cb = cb, ca
                if ca < tau_index * cb:
                    continue
                if jaccard(features_of(a), features_of(b)) >= threshold:
                    uf.union(a, b)
        # 空文档：特征集为空，两篇空文档相似度定义为 1，彼此成组；
        # 空文档没有前缀 token，不会与任何非空文档成为候选（相似度为 0）。
        for other in empty_ids[1:]:
            uf.union(empty_ids[0], other)
        return uf

    # ---- 输出：代表 = 组内最小编号，组内升序，组间按代表升序，只留 >=2 人的组 ----

    @staticmethod
    def _format_groups(uf):
        groups = [sorted(members) for members in uf.components() if len(members) >= 2]
        groups.sort(key=lambda g: g[0])
        return groups
