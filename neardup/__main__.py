"""命令行入口。

    python -m neardup run docs.jsonl -o groups.txt [--threshold 0.8]
    python -m neardup evaluate labels.csv groups.txt
"""

import argparse
import sys
import time

from .evaluate import evaluate
from .pipeline import Config, find_duplicate_groups


def main(argv=None):
    parser = argparse.ArgumentParser(prog="neardup")
    sub = parser.add_subparsers(dest="command", required=True)

    run = subparsers = sub.add_parser("run", help="跑近重复检测，输出分组")
    run.add_argument("input", help="输入 jsonl，每行 {\"id\":N,\"text\":\"...\"}")
    run.add_argument("-o", "--output", help="分组输出文件，缺省写到标准输出")
    run.add_argument("--threshold", type=float, default=0.8,
                     help="相似度阈值 τ，默认 0.8")
    run.add_argument("--temp-dir", default=None, help="临时文件目录")
    run.add_argument("--chunk-records", type=int, default=1_000_000,
                     help="外排序每个分块的记录数，调小可压内存")

    ev = sub.add_parser("evaluate", help="在标注对上评估召回与误报")
    ev.add_argument("labels", help="标注 csv：id_a,id_b,label")
    ev.add_argument("groups", help="run 产出的分组文件")

    args = parser.parse_args(argv)
    if args.command == "run":
        config = Config(threshold=args.threshold, temp_dir=args.temp_dir,
                        chunk_records=args.chunk_records)
        started = time.perf_counter()
        if args.output:
            stats = find_duplicate_groups(args.input, args.output, config)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(
                    mode="r", suffix=".txt") as tmp:
                stats = find_duplicate_groups(args.input, tmp.name, config)
                with open(tmp.name, encoding="utf-8") as f:
                    sys.stdout.write(f.read())
        elapsed = time.perf_counter() - started
        print("docs=%d exact_dup_unions=%d candidates=%d verified=%d "
              "merged=%d groups=%d elapsed=%.1fs"
              % (stats.n_docs, stats.n_exact_dup_unions,
                 stats.n_candidate_pairs, stats.n_verified_pairs,
                 stats.n_merged_pairs, stats.n_groups, elapsed),
              file=sys.stderr)
    else:
        recall, fp_rate, detail = evaluate(args.labels, args.groups)
        print("recall=%.4f (%d/%d)  false_positive=%.4f (%d/%d)"
              % (recall, detail["positives_hit"], detail["positives"],
                 fp_rate, detail["negatives_wrongly_grouped"],
                 detail["negatives"]))


if __name__ == "__main__":
    main()
