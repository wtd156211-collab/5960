import unittest

from neardup import extract_features, jaccard, normalize


class TestNormalize(unittest.TestCase):
    def test_removes_all_whitespace(self):
        self.assertEqual(normalize("a b\tc\nd\re\x0bf\x0cg h　i"), "abcdefghi")

    def test_ascii_letters_lowercased(self):
        self.assertEqual(normalize("AbC XYZ"), "abc xyz".replace(" ", ""))

    def test_non_ascii_and_punctuation_preserved(self):
        # 全角字母不转小写，标点（含全角）原样保留
        self.assertEqual(normalize("ＡＢ，。C！"), "ＡＢ，。c！")

    def test_empty(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(" \t\n"), "")


class TestFeatures(unittest.TestCase):
    def test_empty_text_empty_features(self):
        self.assertEqual(extract_features(""), frozenset())

    def test_short_text_is_single_feature(self):
        self.assertEqual(extract_features("去重"), frozenset({"去重"}))
        self.assertEqual(extract_features("x"), frozenset({"x"}))

    def test_gram_set(self):
        self.assertEqual(extract_features("abcd"),
                         frozenset({"abc", "bcd"}))

    def test_duplicate_grams_counted_once(self):
        self.assertEqual(extract_features("aaaa"), frozenset({"aaa"}))


class TestJaccard(unittest.TestCase):
    def test_two_empty_sets_are_identical(self):
        self.assertEqual(jaccard(frozenset(), frozenset()), 1.0)

    def test_empty_vs_nonempty(self):
        self.assertEqual(jaccard(frozenset(), frozenset({"a"})), 0.0)
        self.assertEqual(jaccard(frozenset({"a"}), frozenset()), 0.0)

    def test_plain(self):
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        self.assertAlmostEqual(jaccard(a, b), 0.5)


if __name__ == "__main__":
    unittest.main()
