import argparse
import sys
from typing import Any, Dict

from src.prepare import run as prepare
from src.train import train, continue_train
from src.predict import predict
from src.export import export_onnx
from src.utils import CFG, ModelNotFoundError, DatasetNotFoundError, log


def main() -> None:
    """主入口"""
    parser = argparse.ArgumentParser(description="YOLO 织物分类流水线")
    subparsers = parser.add_subparsers(required=True)

    # prepare
    cmd = subparsers.add_parser("prepare", help="数据准备")
    cmd.add_argument("dataset", nargs="?", default="ibug")
    cmd.set_defaults(fn=lambda a: prepare(a.dataset))

    # train
    cmd = subparsers.add_parser("train", help="模型训练")
    cmd.add_argument("--data", required=True)
    cmd.add_argument("--model")
    cmd.add_argument("--epochs", type=int)
    cmd.add_argument("--imgsz", type=int)
    cmd.add_argument("--batch", type=int)
    cmd.add_argument("--lr", type=float)
    cmd.add_argument("--device")
    cmd.add_argument("--workers", type=int)
    cmd.add_argument("--patience", type=int)
    cmd.add_argument("--resume", action="store_true")
    cmd.set_defaults(fn=lambda a: train(
        **{k: v for k, v in vars(a).items() if k != "fn" and v is not None},
        train_cfg=CFG["train"]
    ))

    # continue
    cmd = subparsers.add_parser("continue", help="继续训练")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--data", required=True)
    cmd.add_argument("--epochs", type=int)
    cmd.add_argument("--imgsz", type=int)
    cmd.add_argument("--batch", type=int)
    cmd.add_argument("--lr", type=float)
    cmd.add_argument("--device")
    cmd.add_argument("--workers", type=int)
    cmd.add_argument("--patience", type=int)
    cmd.set_defaults(fn=lambda a: continue_train(
        **{k: v for k, v in vars(a).items() if k != "fn" and v is not None},
        train_cfg=CFG["train"]
    ))

    # predict
    cmd = subparsers.add_parser("predict", help="模型推理")
    cmd.add_argument("--input_path", required=True)
    cmd.add_argument("--model")
    cmd.add_argument("--imgsz", type=int)
    cmd.add_argument("--device")
    cmd.add_argument("--top", type=int)
    cmd.add_argument("--threshold", type=float)
    cmd.add_argument("--show", action="store_true")
    cmd.add_argument("--save", action="store_true")
    cmd.set_defaults(fn=lambda a: predict(
        **{k: v for k, v in vars(a).items() if k != "fn" and v is not None},
        predict_cfg=CFG["predict"]
    ))

    # export
    cmd = subparsers.add_parser("export", help="模型导出")
    cmd.add_argument("--model", nargs="+", required=True)
    cmd.add_argument("--imgsz", type=int, default=384)
    cmd.add_argument("--half", action="store_true")
    cmd.set_defaults(fn=lambda a: [export_onnx(m, a.imgsz, a.half) for m in a.model])

    try:
        args = parser.parse_args()
        args.fn(args)
    except (ModelNotFoundError, DatasetNotFoundError, FileNotFoundError) as e:
        log.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("用户中断")
        sys.exit(0)
    except Exception as e:
        log.error("执行失败: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
