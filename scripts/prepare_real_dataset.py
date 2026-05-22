from pathlib import Path
import random
import shutil
import argparse

CLASSES = ["invoice", "letter", "memo", "resume", "scientific_report"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def copy_split(files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, out_dir / src.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/real_raw/docs-sm")
    parser.add_argument("--output", default="data/real")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    random.seed(args.seed)

    if output.exists():
        shutil.rmtree(output)

    for class_name in CLASSES:
        class_dir = source / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        if len(files) < 10:
            raise ValueError(f"Not enough images for class {class_name}: {len(files)}")

        random.shuffle(files)

        n = len(files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        copy_split(train_files, output / "train" / class_name)
        copy_split(val_files, output / "val" / class_name)
        copy_split(test_files, output / "test" / class_name)

        print(
            f"{class_name}: "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

    print(f"\nPrepared real dataset at: {output}")


if __name__ == "__main__":
    main()