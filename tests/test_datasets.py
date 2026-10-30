"""2차 실험 데이터(pill_detection.datasets)의 이미지 선택, 목록 CSV, 학습 폴더 만들기를 본다.

- 장난감 데이터: 선택 규칙(조합 단위, 상한, 결정성)과 방법별 규칙, YOLO·COCO 형식을 몇 초 안에 본다.
- 실제 데이터: 01 노트북의 선택 결과를 그대로 재현하는지 본다. 데이터가 없으면 건너뛴다.
"""

import csv
import json
import random

import pytest
import yaml

from conftest import skip_without
from helpers import (
    AIHUB_TARGETS_CSV,
    DATA_DIR,
    EXPECTED_RENAMES,
    N_CLASSES,
    N_COMPETITION,
    SELECT_ONLY_COMPETITION,
    SELECT_OTHERS,
    SELECT_V2_AIHUB,
    SELECT_V3_AIHUB,
    TEST_CROPS_CSV,
    V3_N_CLASSES,
    VARIANT_N_CLASSES,
)

from pill_detection import datasets as ds
from pill_detection.utils import clip_xywh, yolo_to_xywh

ANGLES = ("70", "75", "90")
BOX = [10, 10, 20, 30]
EDGE_BOX = [90, 90, 20, 20]  # 100×100 이미지 밖으로 나간 box. [90, 90, 10, 10]으로 잘린다
# (출처, 조합의 클래스). 조합마다 각도 3장. 대회 클래스는 Kaggle 1, 2, 3 + test 전용 5. 9, 10, 11은 AI Hub에만 있다
TOY_COMBOS = [
    ("kaggle", (1, 2)),  # 1차 train
    ("kaggle", (1, 3)),  # 1차 val
    ("aihub", (1, 5)),
    ("aihub", (2, 5)),
    ("aihub", (3, 5)),
    ("aihub", (5, 9)),
    ("aihub", (3, 9, 10)),
    ("aihub", (9, 10)),
    ("aihub", (9, 11)),
]
TOY_COMPETITION = {1, 2, 3, 5}
TOY_TEST = {1: 1, 2: 1, 3: 1, 5: 3}
TOY_VAL_ARGS = {"share": 0.5, "cap": 0.5, "floor": 1}


def combo_of(classes):
    return "K-" + "-".join(f"{c:06d}" for c in classes)


def toy_images():
    """TOY_COMBOS의 이미지 정보. load_images()의 결과와 같은 모양이다."""
    images, ann_id = {}, 1
    for image_id, (source, classes, angle) in enumerate(
            ((s, cs, a) for s, cs in TOY_COMBOS for a in ANGLES), start=1):
        name = f"{combo_of(classes)}_0_2_0_2_{angle}_000_200.png"
        objects = []
        for i, c in enumerate(classes):
            objects.append((c, EDGE_BOX if (c == 9 and angle == "90") else BOX, ann_id))
            ann_id += 1
        images[name] = {"source": source, "file_name": name, "combo": combo_of(classes), "image_id": image_id,
                        "width": 100, "height": 100, "objects": objects}
    return dict(sorted(images.items()))


def toy_split(images):
    """1차 분할: Kaggle (1, 2) 조합은 train, (1, 3) 조합은 val."""
    return {name: "val" if image["combo"] == combo_of((1, 3)) else "train"
            for name, image in images.items() if image["source"] == "kaggle"}


def write_toy_data(data_dir, images):
    """장난감 이미지를 정제한 학습 데이터와 같은 폴더 구조(Kaggle JSON, PNG)로 쓴다."""
    for name, image in images.items():
        source = image["source"]
        png = data_dir / ds.IMAGE_DIRS[source] / name
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(name.encode())
        for c, bbox, ann_id in image["objects"]:
            path = data_dir / ds.ANN_DIRS[source] / f"{image['combo']}_json" / f"K-{c:06d}" / name.replace(".png", ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "images": [{"file_name": name, "width": 100, "height": 100, "id": image["image_id"]}],
                "annotations": [{"area": bbox[2] * bbox[3], "iscrowd": 0, "bbox": bbox, "category_id": c, "ignore": 0,
                                 "segmentation": [], "id": ann_id, "image_id": image["image_id"]}],
                "categories": [{"supercategory": "pill", "id": c, "name": f"약{c} "}],
            }, ensure_ascii=False), encoding="utf-8")


def toy_lists(images, val_args=TOY_VAL_ARGS):
    """장난감 데이터로 make_dataset_lists.py와 같은 순서로 val, 방법별 train, CSV 줄을 만든다."""
    split = toy_split(images)
    val = ds.select_val(images, TOY_COMPETITION, TOY_TEST, split, **val_args)["files"]
    variants = ds.select_variants(images, val, TOY_COMPETITION, split, target=3)
    names = {c: f"약{c}" for image in images.values() for c in ds.object_classes(image)}
    rows = ds.make_image_rows(images, val, variants, split, TOY_COMPETITION)
    table = ds.build_class_table(images, names, val, variants, TOY_COMPETITION, TOY_TEST)
    return val, variants, rows, table


def test_plan_uniform_prefers_images_with_fewer_filled_classes():
    """목표를 이미 채운 클래스가 적게 함께 찍힌 이미지부터 넣고, 목표에 닿으면 멈춘다."""
    img = lambda name, classes: {"file_name": name, "objects": [(c, BOX, 0) for c in classes]}  # noqa: E731
    base = [img("base", [3])]
    chosen, count = ds.plan_uniform(base, [img("A", [1, 3]), img("B", [1]), img("C", [2])], target=1)
    assert chosen == ["B", "C"]
    assert count == {1: 1, 2: 1, 3: 1}
    chosen, _ = ds.plan_uniform([], [img("A", [1, 2]), img("B", [1]), img("C", [2])], target=1)
    assert chosen == ["A"]  # A 하나로 1과 2가 모두 목표를 채운다


def test_select_val_rules():
    """1차 val Kaggle 조합을 넣고, AI Hub는 조합 통째로, 상한을 지키며, 대회 클래스만 든 조합부터 고른다."""
    images = toy_images()
    plan = ds.select_val(images, TOY_COMPETITION, TOY_TEST, toy_split(images), **TOY_VAL_ARGS)
    combos = {images[name]["combo"] for name in plan["files"]}
    assert combos == {combo_of((1, 3)), combo_of((3, 5)), combo_of((5, 9))}
    assert all(image["combo"] not in combos or name in plan["files"] for name, image in images.items())  # 조합 단위
    assert plan["target"] == {1: 1, 2: 1, 3: 2, 5: 6}
    assert all(n <= plan["cap"][c] for c, n in plan["aihub_count"].items())
    assert plan["capped"] == {1, 2}  # (1, 5), (2, 5) 조합은 1, 2의 상한(1개)을 넘겨서 건너뛴다
    assert plan["count"][2] == 0  # 그래서 2는 val에 없다 (보고표에 적는 경우)


def test_select_val_is_deterministic():
    """입력 순서를 섞어도 같은 val이 나온다."""
    images = toy_images()
    expected = ds.select_val(images, TOY_COMPETITION, TOY_TEST, toy_split(images), **TOY_VAL_ARGS)["files"]
    items = list(images.items())
    random.Random(0).shuffle(items)
    assert ds.select_val(dict(items), TOY_COMPETITION, TOY_TEST, toy_split(images), **TOY_VAL_ARGS)["files"] == expected


def test_variant_rules_and_model_classes():
    """방법끼리 val이 같고, train과 val이 겹치지 않고, v2·v3 ⊆ v1 = v4이다. 모델 클래스는 train에 객체가 있는 클래스다."""
    images = toy_images()
    val, variants, rows, _ = toy_lists(images)
    kaggle_train = {name for name, s in toy_split(images).items() if s == "train"}
    for v in ds.VARIANTS:
        assert not variants[v] & val
        assert kaggle_train <= variants[v]
        assert {r["file_name"] for r in rows if r[v] and r["split"] == "val"} == val
    assert variants["v2"] <= variants["v1"] and variants["v3"] <= variants["v1"] and variants["v4"] == variants["v1"]
    assert {images[n]["combo"] for n in variants["v1"] - variants["v3"]} == {combo_of((9, 10)), combo_of((9, 11))}
    assert ds.model_classes(images, variants["v1"], "v1", TOY_COMPETITION) == [1, 2, 3, 5, 9, 10, 11]
    assert ds.model_classes(images, variants["v3"], "v3", TOY_COMPETITION) == [1, 2, 3, 5, 9, 10]
    assert ds.model_classes(images, variants["v4"], "v4", TOY_COMPETITION) == [1, 2, 3, 5, ds.OTHERS_CATEGORY_ID]


def test_image_rows_and_class_table():
    """images.csv와 classes.csv의 열과 값: 출처 순서, 대회 클래스 수, 그룹, v4의 others 합치기."""
    images = toy_images()
    val, variants, rows, table = toy_lists(images)
    assert list(rows[0]) == ds.IMAGE_COLUMNS
    assert [r["source"] for r in rows] == sorted((r["source"] for r in rows), key=ds.SOURCES.index)
    mixed = next(r for r in rows if r["combo"] == combo_of((3, 9, 10)))
    assert (mixed["n_objects"], mixed["n_competition"], mixed["n_noncompetition"]) == (3, 1, 2)
    assert all(r["exp1_split"] == "" for r in rows if r["source"] == "aihub")
    by_id = {r["category_id"]: r for r in table}
    assert list(table[0]) == ds.CLASS_COLUMNS
    assert [by_id[c]["group"] for c in (1, 5, 9, ds.OTHERS_CATEGORY_ID)] == ["kaggle", "test_only", "aihub_only", "others"]
    assert by_id[11]["idx_v3"] == "" and by_id[11]["idx_v4"] == "" and by_id[ds.OTHERS_CATEGORY_ID]["idx_v4"] == 4
    others = by_id[ds.OTHERS_CATEGORY_ID]
    assert others["n_train_v4"] == sum(by_id[c]["n_train_v1"] for c in (9, 10, 11))
    assert by_id[9]["n_train_v4"] == 0 and others["n_train_v1"] == ""
    assert others["n_val"] == sum(by_id[c]["n_val"] for c in (9, 10, 11))


def test_write_rows_reports_unchanged(tmp_path):
    """같은 내용으로 다시 쓰면 파일을 건드리지 않고 False를 돌려준다."""
    path = tmp_path / "images.csv"
    rows = [{"a": 1, "b": "x"}]
    assert ds.write_rows(path, ["a", "b"], rows) is True
    assert ds.write_rows(path, ["a", "b"], rows) is False
    assert ds.write_rows(path, ["a", "b"], [{"a": 2, "b": "x"}]) is True
    assert ds.read_rows(path) == [{"a": "2", "b": "x"}]


@pytest.fixture
def toy_dataset(tmp_path):
    """장난감 학습 데이터 폴더와 목록 CSV 2개."""
    images = toy_images()
    data_dir = tmp_path / "data"
    write_toy_data(data_dir, images)
    _, _, rows, table = toy_lists(ds.load_images(data_dir)[0])
    images_csv, classes_csv = tmp_path / "images.csv", tmp_path / "classes.csv"
    ds.write_rows(images_csv, ds.IMAGE_COLUMNS, rows)
    ds.write_rows(classes_csv, ds.CLASS_COLUMNS, table)
    return {"images": images, "data_dir": data_dir, "images_csv": images_csv, "classes_csv": classes_csv,
            "out": tmp_path / "out"}


def test_load_images_reads_toy_data(toy_dataset):
    """load_images는 쓴 그대로의 이미지와 객체를 읽고, 제품명 끝의 공백을 뗀다."""
    images, names = ds.load_images(toy_dataset["data_dir"])
    assert images == toy_dataset["images"]
    assert names[5] == "약5"


def test_build_dataset_yolo_and_coco(toy_dataset):
    """YOLO 라벨은 되돌리면 잘라낸 box와 같고, data.yaml의 이름은 K-코드다. COCO와 채점 정답도 맞다."""
    t = toy_dataset
    info = ds.build_dataset("v4", t["out"] / "v4", t["data_dir"], t["images_csv"], t["classes_csv"])
    out = t["out"] / "v4"
    data_yaml = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["nc"] == info["nc"] == 5
    assert data_yaml["names"] == {0: "1", 1: "2", 2: "3", 3: "5", 4: "others"}
    lists = ds.read_image_list(t["images_csv"], "v4")
    assert sorted(p.name for p in (out / "images" / "val").iterdir()) == sorted(lists["val"])
    for split, names in lists.items():
        for name in names:
            image = t["images"][name]
            lines = (out / "labels" / split / name.replace(".png", ".txt")).read_text().splitlines()
            assert len(lines) == len(image["objects"])
            for line, (c, bbox, _) in zip(lines, image["objects"]):
                idx, *yolo = line.split()
                assert data_yaml["names"][int(idx)] == ds.category_label(ds.variant_category(c, "v4", TOY_COMPETITION))
                assert yolo_to_xywh([float(v) for v in yolo], 100, 100) == pytest.approx(clip_xywh(bbox, 100, 100), abs=1e-3)
    train, val = (json.loads((out / "coco" / f"{s}.json").read_text(encoding="utf-8")) for s in ("train", "val"))
    assert train["categories"] == val["categories"]
    assert [c["id"] for c in train["categories"]] == [1, 2, 3, 5, ds.OTHERS_CATEGORY_ID]
    gt = json.loads((out / "eval" / "val_gt.json").read_text(encoding="utf-8"))
    assert {a["category_id"] for a in gt["annotations"]} == {c for n in lists["val"] for c, _, _ in t["images"][n]["objects"]}
    assert {c["id"]: c["competition"] for c in gt["categories"]}[9] == 0
    assert info["dropped_objects"] == {"train": 0, "val": 0}
    assert (out / "classes.csv").read_bytes() == t["classes_csv"].read_bytes()


def test_build_dataset_same_val_for_all_variants(toy_dataset):
    """val 이미지와 채점 정답은 4가지 방법이 같다. v3는 train에 없는 클래스의 val 객체를 YOLO 라벨에서 뺀다."""
    t = toy_dataset
    infos = {v: ds.build_dataset(v, t["out"] / v, t["data_dir"], t["images_csv"], t["classes_csv"]) for v in ds.VARIANTS}
    gts = {(t["out"] / v / "eval" / "val_gt.json").read_bytes() for v in ds.VARIANTS}
    assert len(gts) == 1
    assert len({tuple(sorted(p.name for p in (t["out"] / v / "images" / "val").iterdir())) for v in ds.VARIANTS}) == 1
    assert infos["v1"]["nc"] == 7 and infos["v3"]["nc"] == 6
    assert infos["v1"]["dropped_objects"]["val"] == 0


def test_build_dataset_refuses_bad_out(toy_dataset):
    """비어 있지 않은 폴더, 학습 데이터 원본 폴더 안에는 쓰지 않는다."""
    t = toy_dataset
    t["out"].mkdir()
    (t["out"] / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="비어 있지 않습니다"):
        ds.build_dataset("v1", t["out"], t["data_dir"], t["images_csv"], t["classes_csv"])
    with pytest.raises(ValueError, match="원본 폴더 안"):
        ds.build_dataset("v1", t["data_dir"] / "raw" / "exp2", t["data_dir"], t["images_csv"], t["classes_csv"])


def test_read_exp1_split_checks(tmp_path):
    """1차 분할 CSV: 한 조합이 train과 val에 나뉘어 있으면 거부한다."""
    path = tmp_path / "split.csv"
    a, b = (f"{combo_of((1, 2))}_0_2_0_2_{angle}_000_200.png" for angle in ("70", "75"))
    path.write_text(f"file_name,combo,split\n{a},x,train\n{b},x,val\n", encoding="utf-8")
    with pytest.raises(ValueError, match="나뉘어"):
        ds.read_exp1_split(path)


def test_reproduces_notebook01_selection():
    """val을 떼기 전 전체 데이터에서 01 노트북의 선택 결과(4-5, 5-4~5-6)를 그대로 낸다."""
    skip_without(DATA_DIR / ds.ANN_DIRS["kaggle"], DATA_DIR / ds.ANN_DIRS["aihub"])
    images, names = ds.load_images(DATA_DIR)
    competition = ds.competition_classes(images, ds.TEST_ONLY_CLASSES)
    assert (len(names), len(competition)) == (N_CLASSES, N_COMPETITION)
    kaggle = [image for image in images.values() if image["source"] == "kaggle"]
    aihub = [image for image in images.values() if image["source"] == "aihub"]

    # 4-5: 노트북은 원래 파일명 순서로 골랐다. 이름이 바뀐 13장은 원래 이름으로 정렬해야 같은 목록이 나온다
    original = {new: old for old, new in EXPECTED_RENAMES.items()}
    chosen, _ = ds.plan_uniform(kaggle, aihub, ds.V2_TARGET, sort_key=lambda name: original.get(name, name))
    assert len(chosen) == SELECT_V2_AIHUB
    if AIHUB_TARGETS_CSV.is_file():
        with open(AIHUB_TARGETS_CSV, newline="", encoding="utf-8") as f:
            assert set(chosen) == {row["export_file_name"] for row in csv.DictReader(f)}

    only = [image for image in aihub if set(ds.object_classes(image)) <= competition]
    assert (len(only), len({image["combo"] for image in only})) == SELECT_ONLY_COMPETITION
    assert ds.object_counts(images, images, "v4", competition)[ds.OTHERS_CATEGORY_ID] == SELECT_OTHERS
    v3 = {image["file_name"] for image in aihub if set(ds.object_classes(image)) & competition}
    assert len(v3) == SELECT_V3_AIHUB
    v3_train = v3 | {image["file_name"] for image in kaggle}
    assert len(ds.model_classes(images, v3_train, "v3", competition)) == V3_N_CLASSES


def test_committed_lists_reproduce():
    """커밋한 data/datasets/의 CSV 2개를 같은 입력(1차 분할은 images.csv의 exp1_split 열)으로 다시 만들면 글자까지 같다."""
    skip_without(DATA_DIR / ds.ANN_DIRS["kaggle"], DATA_DIR / ds.ANN_DIRS["aihub"])
    images_csv, classes_csv = DATA_DIR / "datasets" / "images.csv", DATA_DIR / "datasets" / "classes.csv"
    images, names = ds.load_images(DATA_DIR)
    result = ds.make_lists(images, names, ds.exp1_split_from_rows(ds.read_rows(images_csv)),
                           ds.read_test_counts(TEST_CROPS_CSV))
    assert ds.rows_to_text(ds.IMAGE_COLUMNS, result["image_rows"]) == images_csv.read_text(encoding="utf-8")
    assert ds.rows_to_text(ds.CLASS_COLUMNS, result["class_rows"]) == classes_csv.read_text(encoding="utf-8")

    val, variants, competition = result["val"], result["variants"], result["competition"]
    assert len(competition) == N_COMPETITION
    n_aihub = sum(image["source"] == "aihub" for image in images.values())
    assert 0.07 <= sum(images[n]["source"] == "aihub" for n in val["files"]) / n_aihub <= 0.12
    assert all(n <= val["cap"][c] for c, n in val["aihub_count"].items())
    assert all(not variants[v] & val["files"] for v in ds.VARIANTS)
    assert {v: len(ds.model_classes(images, variants[v], v, competition)) for v in ds.VARIANTS} == VARIANT_N_CLASSES
