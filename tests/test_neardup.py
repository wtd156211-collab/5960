import csv
import io
import json
import os
import tempfile
import tracemalloc
import unittest

from neardup import Detector, detect, extract_features, jaccard, normalize_text
from neardup.detect import prefix_length
from neardup.externalsort import external_sort
from neardup.unionfind import UnionFind

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "samples", "docs.jsonl")
LABELS = os.path.join(ROOT, "samples", "labels.csv")
EXPECTED = os.path.join(ROOT, "samples", "expected-groups.txt")


class TestNormalize(unittest.TestCase):
    def test_strips_all_whitespace(self):
        self.assertEqual(normalize_text(" a b\tc\nd\re\vf\fg "), "abcdefg")

    def test_ascii_letters_lowercased(self):
        self.assertEqual(normalize_text("AbC XYZ"), "abc xyz".replace(" ", ""))

    def test_punctuation_and_non_ascii_kept(self):
        # 标点保留（改动标点算改动）；非 ASCII 字母不做大小写转换
        self.assertEqual(normalize_text("【转载】你好，World！"), "【转载】你好，world！")

    def test_empty(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("  \n\t "), "")


class TestFeatures(unittest.TestCase):
    def test_3gram_set(self):
        self.assertEqual(extract_features("abcd"), frozenset({"abc", "bcd"}))

    def test_duplicate_grams_counted_once(self):
        self.assertEqual(extract_features("aaaa"), frozenset({"aaa"}))

    def test_short_string_is_single_feature(self):
        self.assertEqual(extract_features("ab"), frozenset({"ab"}))
        self.assertEqual(extract_features("x"), frozenset({"x"}))

    def test_empty_is_empty_set(self):
        self.assertEqual(extract_features(""), frozenset())


class TestJaccard(unittest.TestCase):
    def test_two_empty_sets_are_identical(self):
        self.assertEqual(jaccard(frozenset(), frozenset()), 1.0)

    def test_empty_vs_nonempty_is_zero(self):
        self.assertEqual(jaccard(frozenset(), frozenset({"a"})), 0.0)
        self.assertEqual(jaccard(frozenset({"a"}), frozenset()), 0.0)

    def test_identical(self):
        s = frozenset({"a", "b"})
        self.assertEqual(jaccard(s, s), 1.0)

    def test_value(self):
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        self.assertAlmostEqual(jaccard(a, b), 2 / 4)


class TestPrefixLength(unittest.TestCase):
    def test_bounds(self):
        for n in (1, 2, 5, 100, 1000):
            p = prefix_length(n, 0.8)
            self.assertGreaterEqual(p, 1)
            self.assertLessEqual(p, n)

    def test_completeness_simulation(self):
        # 对任意满足 Jaccard >= tau 的两个集合，df 序前缀必然相交
        import random
        rng = random.Random(42)
        tau = 0.8
        df = {t: rng.randint(1, 100) for t in range(500)}
        for _ in range(2000):
            union = rng.sample(range(500), 60)
            inter = rng.sample(union, 48)  # J = 48/60 = 0.8
            a = set(rng.sample(union, 50))
            b = set(union) - a | set(inter)
            b = set(rng.sample(sorted(b), min(len(b), 52)))
            if jaccard(frozenset(a), frozenset(b)) < tau:
                continue
            key = lambda t: (df[t], t)
            pa = sorted(a, key=key)[:prefix_length(len(a), tau)]
            pb = sorted(b, key=key)[:prefix_length(len(b), tau)]
            self.assertTrue(set(pa) & set(pb))


class TestExternalSort(unittest.TestCase):
    def test_sorted_unique_across_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.txt")
            dst = os.path.join(tmp, "out.txt")
            lines = ["b\t1\n", "a\t2\n", "b\t1\n", "c\t0\n", "a\t2\n"] * 500
            with open(src, "w") as f:
                f.writelines(lines)
            # 极小的块 + 多轮归并也要给出正确结果
            external_sort(src, dst, tmp, chunk_lines=7, unique=True)
            with open(dst) as f:
                got = f.read()
            self.assertEqual(got, "a\t2\nb\t1\nc\t0\n")

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.txt")
            dst = os.path.join(tmp, "out.txt")
            open(src, "w").close()
            external_sort(src, dst, tmp, chunk_lines=10)
            with open(dst) as f:
                self.assertEqual(f.read(), "")


class TestUnionFind(unittest.TestCase):
    def test_components(self):
        uf = UnionFind()
        uf.union(3, 1)
        uf.union(1, 2)
        uf.union(8, 9)
        comps = sorted(sorted(c) for c in uf.components())
        self.assertEqual(comps, [[1, 2, 3], [8, 9]])


def _write_docs(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")


class TestEdgeCases(unittest.TestCase):
    """README「边界情况」一节的全部口径。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        docs = [
            (1, ""),                      # 空文档对：相似度定义为 1，彼此成组
            (2, "   \n\t"),
            (3, "你好"),                  # 两字文档：完全相同才成组
            (4, "你 好"),                 # 归一化后与 3 相同
            (5, "你好啊"),                # 只差一个字：不成组
            (6, "完全一样的内容，一字不差。"),   # 完全重复：必然成组
            (7, "完全一样的内容，一字不差。"),
            (8, "订单A今日已发货，请注意查收。"),   # 模板相近、主体词不同：不成组
            (9, "订单B今日已发货，请注意查收。"),
            (10, "订单C今日已发货，请注意查收。"),
            (11, "一篇孤零零的、和谁都不一样的内容。"),  # 单篇：不占行
        ]
        cls.docs_path = os.path.join(cls.tmp.name, "docs.jsonl")
        _write_docs(cls.docs_path, docs)
        cls.groups = detect(cls.docs_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_groups(self):
        self.assertEqual(self.groups, [[1, 2], [3, 4], [6, 7]])

    def test_template_lookalikes_not_grouped(self):
        for g in self.groups:
            self.assertFalse({8, 9, 10} & set(g))

    def test_singleton_not_output(self):
        self.assertNotIn([11], self.groups)

    def test_threshold_configurable(self):
        # 阈值拉到 1.0 时，只有完全一样的才成组
        groups = Detector(threshold=1.0).detect(self.docs_path)
        self.assertEqual(groups, [[1, 2], [3, 4], [6, 7]])
        # 阈值 1.0 时空文档对仍成组（空 vs 空相似度定义为 1）
        self.assertIn([1, 2], groups)

    def test_determinism_and_chunk_independence(self):
        # 结果只与数据有关：跑两遍、换分块参数，输出逐字节一样
        outs = []
        for cfg in (dict(), dict(sort_chunk_lines=3, cache_size=2)):
            out = os.path.join(self.tmp.name, "out.txt")
            Detector(**cfg).detect(self.docs_path, out)
            with open(out, "rb") as f:
                outs.append(f.read())
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], b"1,2\n3,4\n6,7\n")


class TestSampleDataset(unittest.TestCase):
    """在 samples 标注数据集上的端到端门槛（只跑一次，结果复用）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out_path = os.path.join(cls.tmp.name, "groups.txt")
        cls.groups = Detector(threshold=0.8).detect(DOCS, cls.out_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_groups_match_expected_byte_for_byte(self):
        with open(self.out_path, "rb") as f:
            got = f.read()
        with open(EXPECTED, "rb") as f:
            expected = f.read()
        self.assertEqual(got, expected)

    def test_output_format_rules(self):
        reps = []
        for g in self.groups:
            self.assertGreaterEqual(len(g), 2)          # 只输出成员数 >= 2 的组
            self.assertEqual(g, sorted(g))              # 组内升序，第一个就是代表
            reps.append(g[0])
        self.assertEqual(reps, sorted(reps))            # 组间按代表升序
        with open(self.out_path, "rb") as f:
            data = f.read()
        self.assertFalse(data.endswith(b"\n\n"))
        for line in data.split(b"\n")[:-1]:
            self.assertNotIn(b"\r", line)               # 行尾 LF

    def test_recall_and_false_positive_gates(self):
        group_of = {}
        for g in self.groups:
            for doc in g:
                group_of[doc] = g[0]
        pos = neg = tp = fp = 0
        seen = {}
        with open(LABELS, newline="") as f:
            for row in csv.DictReader(f):
                pair = (int(row["id_a"]), int(row["id_b"]))
                label = int(row["label"])
                # 标注噪声：同一对同时标了 1 和 0 时以正例为准。
                # （数据集中仅 (201,202) 一对冲突，其精确 Jaccard = 0.98，
                #  确为近重复且在 expected-groups 的同组里。）
                if pair in seen and seen[pair] != label:
                    seen[pair] = 1
                else:
                    seen[pair] = max(seen.get(pair, 0), label)
        for pair, label in seen.items():
            a, b = pair
            same = group_of.get(a) is not None and group_of.get(a) == group_of.get(b)
            if label == 1:
                pos += 1
                tp += same
            else:
                neg += 1
                fp += same
        recall = tp / pos
        self.assertGreaterEqual(recall, 0.95, "召回不达标")
        self.assertEqual(fp, 0, "误报必须是硬零")

    def test_determinism_rerun_byte_identical(self):
        out2 = os.path.join(self.tmp.name, "groups2.txt")
        Detector(threshold=0.8).detect(DOCS, out2)
        with open(self.out_path, "rb") as f:
            first = f.read()
        with open(out2, "rb") as f:
            self.assertEqual(first, f.read())


class TestMemoryBounded(unittest.TestCase):
    """内存峰值必须可测且可控：分块参数调小，峰值随之下降且不超预算。"""

    def _make_docs(self, path, n=800):
        # 确定性地构造一批「长文本 + 部分近重复」的文档
        import random
        rng = random.Random(7)
        words = ["内容库", "转载", "召回", "误报", "阈值", "指纹", "分块",
                 "归并", "特征", "相似度", "文档", "流式", "临时文件", "缓存"]
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                text = "，".join(rng.choice(words) for _ in range(120))
                if i % 10 == 0 and i > 0:
                    text = text + "【转载】"  # 制造一些近重复
                f.write(json.dumps({"id": i + 1, "text": text},
                                   ensure_ascii=False) + "\n")

    def test_peak_is_bounded_and_follows_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = os.path.join(tmp, "docs.jsonl")
            self._make_docs(docs)
            peaks = {}
            for name, cfg in (("small", dict(sort_chunk_lines=2_000, cache_size=64)),
                              ("large", dict(sort_chunk_lines=200_000, cache_size=4096))):
                tracemalloc.start()
                Detector(**cfg).detect(docs, os.path.join(tmp, "out_%s.txt" % name), tmp)
                peaks[name] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            budget = 512 * 2 ** 20  # 生产预算 512 MB，样例规模必须远低于此
            self.assertLess(peaks["small"], 64 * 2 ** 20)
            self.assertLess(peaks["large"], budget)
            self.assertLess(peaks["small"], peaks["large"])
            # 两种分块参数下结果必须一致（结果只与数据有关）
            with open(os.path.join(tmp, "out_small.txt"), "rb") as f:
                small = f.read()
            with open(os.path.join(tmp, "out_large.txt"), "rb") as f:
                self.assertEqual(small, f.read())


class TestCli(unittest.TestCase):
    def test_main(self):
        from neardup.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            docs = os.path.join(tmp, "docs.jsonl")
            _write_docs(docs, [(1, "完全相同的内容。"), (2, "完全相同的内容。"),
                               (3, "毫不相干的另一篇。")])
            out = os.path.join(tmp, "groups.txt")
            rc = main([docs, "-o", out, "--threshold", "0.8"])
            self.assertEqual(rc, 0)
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"1,2\n")


if __name__ == "__main__":
    unittest.main()
