"""内存硬约束测试：用 tracemalloc 实测峰值，并验证峰值不随
文档数量线性增长（分块流式处理的核心性质）。"""

import json
import os
import random
import tempfile
import tracemalloc
import unittest

from neardup import Config, find_duplicate_groups

# 全部文档的特征集若堆进内存，每篇约 125 个 3-gram、每个 Python
# 对象几十字节，8000 篇就是上百 MB；流式实现的峰值应远低于此。
PEAK_LIMIT_BYTES = 64 * 1024 * 1024
# 外排序分块调小，让两个规模都真正走"多块 + 归并"路径
CHUNK_RECORDS = 50_000


def _make_corpus(path, n_docs, seed):
    rng = random.Random(seed)
    phrases = ["".join(rng.choice("的一是在不了有和人这中大为上个国")
               for _ in range(4)) for _ in range(200)]
    docs = []
    for doc_id in range(1, n_docs + 1):
        text = "，".join(rng.choice(phrases) for _ in range(25))
        docs.append(text)
    for doc_id in range(1, n_docs + 1, 20):
        docs[doc_id % n_docs] = docs[doc_id - 1]          # 完全重复
        docs[(doc_id + 1) % n_docs] = "【转载】" + docs[doc_id - 1]  # 转载
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in enumerate(docs, 1):
            f.write(json.dumps({"id": doc_id, "text": text},
                               ensure_ascii=False))
            f.write("\n")


def _peak_of_run(input_path, output_path):
    config = Config(chunk_records=CHUNK_RECORDS)
    tracemalloc.start()
    try:
        find_duplicate_groups(input_path, output_path, config)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


class MemoryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.peaks = {}
        for n_docs in (4_000, 8_000):
            corpus = os.path.join(cls.dir, "corpus_%d.jsonl" % n_docs)
            _make_corpus(corpus, n_docs, seed=n_docs)
            cls.peaks[n_docs] = _peak_of_run(
                corpus, os.path.join(cls.dir, "groups_%d.txt" % n_docs))

    def test_peak_under_limit(self):
        for n_docs, peak in self.peaks.items():
            self.assertLess(
                peak, PEAK_LIMIT_BYTES,
                "%d 篇文档峰值 %.1f MB 超限" % (n_docs, peak / 2**20))

    def test_peak_does_not_grow_linearly_with_corpus(self):
        small = self.peaks[4_000]
        large = self.peaks[8_000]
        self.assertLess(
            large, small * 2.0,
            "文档数翻倍，峰值 %.1f MB -> %.1f MB，接近线性增长"
            % (small / 2**20, large / 2**20))


if __name__ == "__main__":
    unittest.main()
