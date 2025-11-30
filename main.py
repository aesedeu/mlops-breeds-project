import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Запуск процессов обучения, тестирования и инференса PyTorch модели"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- TRAIN ---
    train_parser = subparsers.add_parser("train", help="Запуск обучения (train.py)")

    # --- TEST ---
    test_parser = subparsers.add_parser("test", help="Запуск тестирования (test.py)")
    test_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к модели", metavar=""
    )

    # --- INFER ---
    infer_parser = subparsers.add_parser("infer", help="Запуск инференса (infer.py)")
    infer_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к модели", metavar=""
    )
    infer_parser.add_argument(
        "--image-path",
        type=str,
        required=False,
        help="Путь к изображению (.jpg)",
        metavar="",
    )

    # --- CONVERT ---
    convert_parser = subparsers.add_parser(
        "convert", help="Запуск конвертации (pt2onnx.py)"
    )
    convert_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к модели", metavar=""
    )
    convert_parser.add_argument(
        "--savepath",
        type=str,
        required=False,
        default="triton/models/image_classifier/1/model.onnx",
        help="Путь сохранения модели",
        metavar="",
    )

    # --- INFER TRITON ---
    infer_triton_parser = subparsers.add_parser(
        "infer-triton", help="Инференс через Triton"
    )
    infer_triton_parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Путь к изображению (.jpg)",
        metavar="",
    )

    args = parser.parse_args()

    # --- ROUTING ---
    if args.mode == "train":
        subprocess.run(
            [
                "uv",
                "run",
                "animal_breeds_classifier/train.py",
            ]
        )

    elif args.mode == "test":
        subprocess.run(
            [
                "uv",
                "run",
                "animal_breeds_classifier/test.py",
                f"test_config.checkpoint='{args.checkpoint}'",
            ]
        )

    elif args.mode == "infer":
        subprocess.run(
            [
                "uv",
                "run",
                "animal_breeds_classifier/infer.py",
                f"test_config.checkpoint='{args.checkpoint}'",
                f"infer_config.file_path={args.image_path}",
            ]
        )
    elif args.mode == "convert":
        subprocess.run(
            [
                "uv",
                "run",
                "animal_breeds_classifier/pt2onnx.py",
                f"test_config.checkpoint='{args.checkpoint}'",
                f"convert.savepath={args.savepath}",
            ]
        )
    elif args.mode == "infer-triton":
        subprocess.run(
            [
                "uv",
                "run",
                "animal_breeds_classifier/infer_triton.py",
                f"infer_config.file_path={args.image_path}",
            ]
        )


if __name__ == "__main__":
    main()
