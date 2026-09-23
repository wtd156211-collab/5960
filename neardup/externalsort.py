"""分块外排序：内存里只放一个块，块落临时文件后多路归并。

用于把「token -> 文档」记录和候选对记录按字典序排序。
排序键就是整行字符串，分组/去重不依赖数值顺序，因此无需零填充
（候选对为了复核时的局部性另行零填充，见 detect.py）。

归并路数有上限：每多一路就要多一个打开的文件，文本模式文件对象
本身有可观的内存开销，路数不设限会让内存峰值随数据量失控。
"""

import heapq
import os
import tempfile

_MERGE_WAYS = 32


def external_sort(input_path, output_path, workdir, chunk_lines=200_000, unique=False):
    """把 input_path 的行排序写入 output_path。unique=True 时去掉重复行。

    任意时刻内存里至多持有 chunk_lines 行，外加至多 _MERGE_WAYS 路
    归并各一行，与输入总量无关。
    """
    chunk_paths = []
    try:
        with open(input_path, "r", encoding="utf-8", newline="") as fin:
            while True:
                chunk = []
                for _ in range(chunk_lines):
                    line = fin.readline()
                    if not line:
                        break
                    chunk.append(line)
                if not chunk:
                    break
                chunk.sort()
                chunk_paths.append(_write_lines(workdir, chunk))

        # 多路归并，路数超限时分多轮，每轮把至多 _MERGE_WAYS 个块并成一个。
        while len(chunk_paths) > 1:
            round_paths = chunk_paths
            chunk_paths = []
            for i in range(0, len(round_paths), _MERGE_WAYS):
                group = round_paths[i:i + _MERGE_WAYS]
                if len(group) == 1:
                    chunk_paths.append(group[0])
                    continue
                merged = _merge_files(group, workdir)
                chunk_paths.append(merged)
                for p in group:
                    os.unlink(p)

        if chunk_paths:
            _merge_files(chunk_paths, workdir, output_path=output_path, unique=unique)
        else:
            open(output_path, "w", encoding="utf-8").close()
    finally:
        for p in chunk_paths:
            if os.path.exists(p):
                os.unlink(p)
    return output_path


def _write_lines(workdir, lines):
    fd, path = tempfile.mkstemp(prefix="chunk_", dir=workdir)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fout:
        fout.writelines(lines)
    return path


def _merge_files(paths, workdir, output_path=None, unique=False):
    """把若干个各自有序的文件归并成一个有序文件。"""
    files = [open(p, "r", encoding="utf-8", newline="") for p in paths]
    try:
        if output_path is None:
            fd, output_path = tempfile.mkstemp(prefix="merge_", dir=workdir)
            fout = os.fdopen(fd, "w", encoding="utf-8", newline="")
        else:
            fout = open(output_path, "w", encoding="utf-8", newline="")
        with fout:
            prev = None
            for line in heapq.merge(*files):
                if unique and line == prev:
                    continue
                fout.write(line)
                prev = line
    finally:
        for f in files:
            f.close()
    return output_path
