"""Hugging Face의 OctoMed/BloodMNIST에서 Neutrophil / Lymphocyte 이미지를 클래스별로 내려받습니다.

사용법:
    pip install datasets pillow
    python download_bloodmnist.py            # 클래스당 30장
    python download_bloodmnist.py --count 50 # 클래스당 50장

결과:
    bloodmnist/neutrophil/neutrophil_000.png ...
    bloodmnist/lymphocyte/lymphocyte_000.png ...
(Teachable Machine에서 클래스별 폴더를 그대로 업로드하면 됩니다.)
"""

import argparse
import os

from datasets import ClassLabel, load_dataset

# MedMNIST BloodMNIST의 표준 라벨 번호
MEDMNIST_LABELS = {
    0: "basophil",
    1: "eosinophil",
    2: "erythroblast",
    3: "immature granulocytes",
    4: "lymphocyte",
    5: "monocyte",
    6: "neutrophil",
    7: "platelet",
}
TARGETS = ["neutrophil", "lymphocyte"]


def find_column(features, kind):
    for name, feature in features.items():
        if kind == "image" and feature.__class__.__name__ == "Image":
            return name
        if kind == "label" and name.lower() in ("label", "labels", "class", "answer"):
            return name
    raise SystemExit(f"{kind} 컬럼을 찾지 못했습니다. 컬럼 목록: {list(features)}")


def label_to_name(value, feature):
    if isinstance(value, list):
        value = value[0]
    if isinstance(feature, ClassLabel):
        return feature.int2str(int(value)).lower()
    if isinstance(value, int):
        return MEDMNIST_LABELS.get(value, str(value))
    return str(value).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30, help="클래스당 이미지 수")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="bloodmnist")
    args = parser.parse_args()

    ds = load_dataset("OctoMed/BloodMNIST", split=args.split, streaming=True)
    features = ds.features
    image_col = find_column(features, "image")
    label_col = find_column(features, "label")
    print(f"image 컬럼: {image_col}, label 컬럼: {label_col}")

    saved = {t: 0 for t in TARGETS}
    for t in TARGETS:
        os.makedirs(os.path.join(args.out, t), exist_ok=True)

    for row in ds:
        name = label_to_name(row[label_col], features[label_col])
        target = next((t for t in TARGETS if t in name), None)
        if target is None or saved[target] >= args.count:
            continue
        path = os.path.join(args.out, target, f"{target}_{saved[target]:03d}.png")
        row[image_col].convert("RGB").save(path)
        saved[target] += 1
        if all(n >= args.count for n in saved.values()):
            break

    print("저장 완료:", saved)
    if any(n < args.count for n in saved.values()):
        print(f"일부 클래스가 부족합니다. --split 을 바꿔 보세요 (예: --split test).")


if __name__ == "__main__":
    main()
