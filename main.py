import argparse
import sys
from typing import Any, Dict

from src.prepare import run as prepare
from src.train import train
from src.predict import predict
from src.export import export_onnx
from src.export_arm import export_onnx_arm
from src.incremental import incremental_train
from src.utils import CFG, ModelNotFoundError, DatasetNotFoundError, log


def main() -> None:
    """主入口"""
    parser = argparse.ArgumentParser(description="YOLO 织物分类流水线")
    subparsers = parser.add_subparsers(required=True)

    # prepare
    cmd = subparsers.add_parser("prepare", help="数据准备")
    cmd.add_argument("dataset", nargs="?", default="yueli")
    cmd.set_defaults(fn=lambda a: prepare(a.dataset))

    # train
    cmd = subparsers.add_parser("train", help="模型训练")
    cmd.add_argument("--data", required=True)
    cmd.add_argument("--model", help="模型名称 (yolo26n / yolo26s-cls.pt 等, 默认: yolo26s-cls.pt)")
    cmd.add_argument("--epochs", type=int)
    cmd.add_argument("--imgsz", type=int)
    cmd.add_argument("--batch", type=int)
    cmd.add_argument("--lr", type=float)
    cmd.add_argument("--device")
    cmd.add_argument("--workers", type=int)
    cmd.add_argument("--patience", type=int)
    cmd.add_argument("--resume", action="store_true")
    cmd.set_defaults(
        fn=lambda a: train(
            **{k: v for k, v in vars(a).items() if k != "fn" and v is not None},
            train_cfg=CFG["train"],
        )
    )

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
    cmd.set_defaults(
        fn=lambda a: predict(
            **{k: v for k, v in vars(a).items() if k != "fn" and v is not None},
            predict_cfg=CFG["predict"],
        )
    )

    # export
    cmd = subparsers.add_parser("export", help="模型导出")
    cmd.add_argument("--model", nargs="+", required=True)
    cmd.add_argument("--imgsz", type=int, default=384)
    cmd.add_argument("--half", action="store_true")
    cmd.set_defaults(fn=lambda a: [export_onnx(m, a.imgsz, a.half) for m in a.model])

    # export-arm
    cmd = subparsers.add_parser("export-arm", help="导出 NCNN 模型 (树莓派)")
    cmd.add_argument("--model", nargs="+", required=True)
    cmd.add_argument("--imgsz", type=int)
    cmd.set_defaults(fn=lambda a: [export_onnx_arm(m, a.imgsz) for m in a.model])

    # incremental
    cmd = subparsers.add_parser("incremental", help="增量训练 (添加新分类不从头训练)")
    cmd.add_argument(
        "--old-model",
        required=True,
        help="旧模型路径 (e.g. runs/classify/d3_yolo26n/weights/best.pt)",
    )
    cmd.add_argument("--data", required=True, help="扩类后的数据目录 (已 prepare)")
    cmd.add_argument(
        "--epochs-head", type=int, default=30, help="阶段1: 训分类头轮数 (default: 30)"
    )
    cmd.add_argument(
        "--epochs-fine",
        type=int,
        default=10,
        help="阶段2: 全模型微调轮数 (0=跳过, default: 10)",
    )
    cmd.add_argument(
        "--lr-head", type=float, default=0.01, help="分类头学习率 (default: 0.01)"
    )
    cmd.add_argument(
        "--lr-fine", type=float, default=0.0005, help="微调学习率 (default: 0.0005)"
    )
    cmd.add_argument("--device")
    cmd.add_argument("--batch", type=int)
    cmd.add_argument("--imgsz", type=int)
    cmd.set_defaults(
        fn=lambda a: incremental_train(
            old_model=a.old_model,
            data=a.data,
            epochs_head=a.epochs_head,
            epochs_fine=a.epochs_fine,
            lr_head=a.lr_head,
            lr_fine=a.lr_fine,
            device=a.device,
            batch=a.batch,
            imgsz=a.imgsz or 384,
        )
    )

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
