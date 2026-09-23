import os
import tempfile
import unittest

from neardup.evaluate import evaluate, load_labels


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_conflicting_labels_positive_wins(self):
        labels = self._write("labels.csv",
                             "id_a,id_b,label\n1,2,1\n1,2,0\n1,3,0\n")
        positives, negatives = load_labels(labels)
        self.assertEqual(positives, {(1, 2)})
        self.assertEqual(negatives, {(1, 3)})

    def test_recall_and_fp(self):
        labels = self._write("labels.csv",
                             "id_a,id_b,label\n1,2,1\n3,4,1\n1,3,0\n")
        groups = self._write("groups.txt", "1,2\n")
        recall, fp_rate, detail = evaluate(labels, groups)
        self.assertEqual(recall, 0.5)
        self.assertEqual(fp_rate, 0.0)
        self.assertEqual(detail["positives"], 2)

    def test_transitive_group_membership(self):
        labels = self._write("labels.csv", "id_a,id_b,label\n1,3,1\n")
        groups = self._write("groups.txt", "1,2,3\n")
        recall, _, _ = evaluate(labels, groups)
        self.assertEqual(recall, 1.0)


if __name__ == "__main__":
    unittest.main()
