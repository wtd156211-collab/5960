import json
import os
import tempfile
import unittest

from neardup import Config, evaluate, find_duplicate_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(REPO_ROOT, "samples")


class SamplesTestCase(unittest.TestCase):
    """在 samples 标注数据集上的端到端门槛测试。"""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.output = os.path.join(cls.dir, "groups.txt")
        stats = find_duplicate_groups(
            os.path.join(SAMPLES, "docs.jsonl"), cls.output)
        cls.stats = stats

    def test_groups_match_expected_byte_for_byte(self):
        with open(self.output, "rb") as f:
            actual = f.read()
        with open(os.path.join(SAMPLES, "expected-groups.txt"), "rb") as f:
            expected = f.read()
        self.assertEqual(actual, expected)

    def test_recall_gate(self):
        recall, _, _ = evaluate(os.path.join(SAMPLES, "labels.csv"),
                                self.output)
        self.assertGreaterEqual(recall, 0.95)

    def test_false_positive_gate(self):
        _, fp_rate, detail = evaluate(os.path.join(SAMPLES, "labels.csv"),
                                      self.output)
        self.assertEqual(fp_rate, 0.0)
        self.assertEqual(detail["negatives_wrongly_grouped"], 0)

    def test_determinism(self):
        second = os.path.join(self.dir, "groups2.txt")
        find_duplicate_groups(os.path.join(SAMPLES, "docs.jsonl"), second)
        with open(self.output, "rb") as f:
            first_bytes = f.read()
        with open(second, "rb") as f:
            self.assertEqual(first_bytes, f.read())

    def test_chunk_size_does_not_change_result(self):
        chunked = os.path.join(self.dir, "groups_chunked.txt")
        find_duplicate_groups(os.path.join(SAMPLES, "docs.jsonl"), chunked,
                              Config(chunk_records=1_000))
        with open(chunked, "rb") as f:
            actual = f.read()
        with open(os.path.join(SAMPLES, "expected-groups.txt"), "rb") as f:
            self.assertEqual(actual, f.read())


class EdgeCasesTestCase(unittest.TestCase):
    """空文档、超短文档、完全重复、模板相近但无关。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _run(self, docs, threshold=0.8):
        input_path = os.path.join(self.dir, "docs.jsonl")
        with open(input_path, "w", encoding="utf-8") as f:
            for doc_id, text in docs:
                f.write(json.dumps({"id": doc_id, "text": text},
                                   ensure_ascii=False))
                f.write("\n")
        output_path = os.path.join(self.dir, "groups.txt")
        find_duplicate_groups(input_path, output_path,
                              Config(threshold=threshold))
        with open(output_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def test_empty_docs_group_together(self):
        groups = self._run([(1, ""), (2, "  \t\n"), (3, "完全不同的内容，"
                                                  "和空文档没有任何关系。")])
        self.assertEqual(groups, ["1,2"])

    def test_short_docs(self):
        groups = self._run([(1, "去重"), (2, "去 重"), (3, "阈值")])
        # 1 和 2 归一化后完全相同；3 只差一个字，不成组
        self.assertEqual(groups, ["1,2"])

    def test_exact_duplicates(self):
        text = "完全相同的两篇文档，相似度必然是 1。"
        groups = self._run([(1, text), (2, text), (3, "另一篇无关的内容。")])
        self.assertEqual(groups, ["1,2"])

    def test_template_similar_but_unrelated(self):
        template = "关于%s的月度报告：本月%s数据平稳，各项指标正常，特此汇总。"
        a = template % ("甲项目", "营收")
        b = template % ("乙项目", "客流")
        c = template % ("甲项目", "营收")
        groups = self._run([(1, a), (2, b), (3, c)])
        self.assertEqual(groups, ["1,3"])

    def test_transitive_closure(self):
        # A~B、B~C 相似，A、C 直接相似度不足阈值，仍应同组
        body = "内容库近重复检测的召回和误报两条线都要看，评审时写清楚。"
        a = body
        b = body[:-4] + "评审时要写清楚。"
        c = b[:-4] + "评审的时候要写清楚。"
        docs = [(1, a), (2, b), (3, c)]
        groups = self._run(docs, threshold=0.7)
        self.assertEqual(groups, ["1,2,3"])

    def test_output_rules(self):
        # 代表取最小编号、成员升序、组间按代表升序、单篇不占行
        text = ("一篇会被转载的文章，内容涉及内容库的近重复检测、"
                "召回与误报两条线，以及评审指标的写法。")
        groups = self._run([(9, text), (3, text), (7, "【转载】" + text),
                            (20, "孤零零的一篇，和谁都不一样。"),
                            (15, "另一篇会被转载的文章，讲的是素材中心的"
                                 "监控指标和人工抽检流程，完全不同。"),
                            (11, "另一篇会被转载的文章，讲的是素材中心的"
                                 "监控指标和人工抽检流程，完全不同。")])
        self.assertEqual(groups, ["3,7,9", "11,15"])


if __name__ == "__main__":
    unittest.main()
