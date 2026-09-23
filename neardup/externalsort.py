"""定长二进制记录的外排序：分块读入、块内排序落盘、多路归并流式输出。

任何时刻内存里只有一个分块（chunk_records 条记录）加上归并堆
（每路一条记录），因此峰值内存与记录总数无关。
"""

import heapq
import os
import tempfile


def sort_records(src_path, dst_path, record_size, tempdir,
                 chunk_records=1_000_000, dedup=False):
    """把 src_path 的定长记录按字节序排序写入 dst_path。

    dedup=True 时归并阶段去掉完全相同的相邻记录。
    记录必须设计成大端字节序，使字节序比较等价于数值比较。
    """
    run_paths = []
    try:
        with open(src_path, "rb") as src:
            while True:
                buf = src.read(record_size * chunk_records)
                if not buf:
                    break
                records = [buf[i:i + record_size]
                           for i in range(0, len(buf), record_size)]
                records.sort()
                fd, path = tempfile.mkstemp(dir=tempdir, prefix="run_")
                with os.fdopen(fd, "wb") as run:
                    run.write(b"".join(records))
                run_paths.append(path)
        _merge_runs(run_paths, dst_path, record_size, dedup)
    finally:
        for path in run_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def _merge_runs(run_paths, dst_path, record_size, dedup):
    with open(dst_path, "wb") as out:
        if not run_paths:
            return
        files = [open(path, "rb") for path in run_paths]
        try:
            heap = []
            for idx, f in enumerate(files):
                record = f.read(record_size)
                if record:
                    heap.append((record, idx))
            heapq.heapify(heap)
            last = None
            while heap:
                record, idx = heapq.heappop(heap)
                if not dedup or record != last:
                    out.write(record)
                    last = record
                nxt = files[idx].read(record_size)
                if nxt:
                    heapq.heappush(heap, (nxt, idx))
        finally:
            for f in files:
                f.close()
