"""归一化文本的磁盘存储：顺序追加写，按文档序号随机读。

内存里只保留每篇文档的偏移量（8 字节）和特征数（4 字节），
几百万篇也只有几十 MB；完整文本始终在磁盘上。
"""

from array import array


class TextStore:
    def __init__(self, path):
        self._path = path
        self._writer = open(path, "wb")
        self.offsets = array("Q", [0])
        self.counts = array("I")
        self._file_pos = 0

    def append(self, normalized, n_features):
        data = normalized.encode("utf-8")
        self._writer.write(data)
        self._file_pos += len(data)
        self.offsets.append(self._file_pos)
        self.counts.append(n_features)

    def finish_writing(self):
        self._writer.close()
        self._reader = open(self._path, "rb")

    def text(self, ordinal):
        self._reader.seek(self.offsets[ordinal])
        length = self.offsets[ordinal + 1] - self.offsets[ordinal]
        return self._reader.read(length).decode("utf-8")

    def count(self, ordinal):
        return self.counts[ordinal]

    def close(self):
        try:
            self._reader.close()
        except AttributeError:
            self._writer.close()
