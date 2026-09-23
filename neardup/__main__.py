"""命令行入口：python -m neardup docs.jsonl -o groups.txt [--threshold 0.8]"""

import argparse
import sys

from .detect import Detector


def main(argv=None):
    parser = argparse.ArgumentParser(prog="neardup", description="内容库近重复检测")
    parser.add_argument("docs", help="输入文档，jsonl，每行 {\"id\":N,\"text\":\"...\"}")
    parser.add_argument("-o", "--output", help="分组输出文件（默认标准输出）")
    parser.add_argument("--threshold", type=float, default=0.8, help="相似度阈值 τ，默认 0.8")
    parser.add_argument("--workdir", help="临时文件目录（默认系统临时目录）")
    args = parser.parse_args(argv)

    groups = Detector(threshold=args.threshold).detect(args.docs, args.output, args.workdir)
    if args.output is None:
        out = sys.stdout
        for g in groups:
            out.write(",".join(map(str, g)) + "\n")
    print("共 %d 组（成员数 >= 2）" % len(groups), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
