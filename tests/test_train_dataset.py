"""학습 데이터 만들기(make_train_dataset)가 01 노트북의 결론을 그대로 지키는지 본다.

- test_run_*: 래퍼를 한 번 실행한 결과. Kaggle 사본, 원본을 건드리지 않았는지, 변경 기록 v2, 다시 실행
- test_aihub_*: AI Hub 결과 폴더. 노트북 2-5(정제 라벨), 2-6(이름 표), 2-7(내보낸 결과), 2-8(다시 점검)의 결론
    - PILL_TEST_AIHUB_EXPORT로 이미 만든 폴더를 줄 수 있다. (conftest.py 참고)
- 같은 장면 검사(2-8 표 C의 RMS)는 이미지 통계가 필요해서 넣지 않았다. 노트북 2-8에서 본다.
"""

import collections
import json
import os
import unicodedata

import pytest

from conftest import skip_without
from helpers import (
    AIHUB_ANN_DIR,
    AIHUB_EXPORT_HASH,
    AIHUB_FIX_ROWS_SHA,
    AIHUB_IMG_DIR,
    AIHUB_N_CLASSES,
    AIHUB_N_COMBOS,
    AIHUB_N_EXCLUDED,
    AIHUB_N_IMAGES,
    AIHUB_N_JSON,
    AIHUB_START_ANN_ID,
    AIHUB_START_IMAGE_ID,
    CHANGE_CSV_V1,
    EXPECTED_RENAMES,
    KAGGLE_ANN_DIR,
    KAGGLE_ANN_KEYS,
    KAGGLE_FIXED_HASH,
    LABEL_FIX_CSV,
    expected_file_name,
    fix_rows_sha,
    pill_codes,
    read_csv_rows,
    read_tree,
    tree_hash,
    write_json_bytes,
)
from pill_detection.data import IMAGE_FIELDS, load_annotations

V2_COLUMNS = ["source", "file_name", "file_name_after", "category_id", "category_id_after", "action",
              "bbox_before", "bbox_after", "area_before", "area_after"]


# ---------- 래퍼 실행 결과 ----------

def test_run_summary(train_dataset):
    """요약 수: Kaggle 763 → 771, AI Hub 이미지 11,974장과 JSON 45,819개, 이름 바뀜 13장, PNG는 모두 하드링크."""
    result = train_dataset["result"]
    assert (result["kaggle"]["n_base"], result["kaggle"]["n_out"]) == (763, 771)
    aihub = result["aihub"]
    assert (aihub["n_images"], aihub["n_json"]) == (AIHUB_N_IMAGES, AIHUB_N_JSON)
    assert aihub["renamed"] == EXPECTED_RENAMES
    assert (aihub["linked"], aihub["copied"]) == (AIHUB_N_IMAGES, 0)


def test_run_kaggle_copy_matches_main(train_dataset):
    """Kaggle 사본은 main 코드가 옛 정제 목록으로 만든 것과 바이트까지 같다."""
    assert tree_hash(train_dataset["kwargs"]["kaggle_out_ann_dir"]) == KAGGLE_FIXED_HASH


def test_run_does_not_touch_raw(train_dataset):
    """원본 annotation 폴더(Kaggle, AI Hub)는 실행 전과 같다."""
    assert tree_hash(KAGGLE_ANN_DIR) == train_dataset["raw_before"]["kaggle"]
    assert tree_hash(AIHUB_ANN_DIR) == train_dataset["raw_before"]["aihub"]


def test_run_change_log_v2(train_dataset):
    """변경 기록은 v1 옆에 apply_label_fixes_v2.csv로 저장되고, Kaggle과 AI Hub의 변경을 함께 담는다."""
    path = train_dataset["result"]["change_csv"]
    assert path.name == "apply_label_fixes_v2.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(V2_COLUMNS)
    rows = read_csv_rows(path)

    # Kaggle 줄은 v1과 같다 (이름과 클래스는 바뀌지 않는다)
    kaggle = [row for row in rows if row["source"] == "kaggle"]
    v1 = read_csv_rows(CHANGE_CSV_V1)
    assert [{key: row[key] for key in v1[0]} for row in kaggle] == v1
    assert all(row["file_name"] == row["file_name_after"] or not row["file_name_after"] for row in kaggle)

    aihub = [row for row in rows if row["source"] == "aihub"]
    # 이름 바뀐 13장은 file_name_after가 새 이름이다
    renamed = {(row["file_name"], row["file_name_after"]) for row in aihub
               if row["file_name_after"] and row["file_name_after"] != row["file_name"]}
    assert renamed == set(EXPECTED_RENAMES.items())
    # 클래스 바꿈 38줄은 category_id_after로 남는다
    assert sum(bool(row["category_id_after"]) and row["category_id_after"] != row["category_id"] for row in aihub) == 38
    # 제외한 이미지 35장의 알약은 delete로 남는다
    excluded = {row["file_name"] for row in read_csv_rows(LABEL_FIX_CSV) if row["source"] == "aihub" and row["action"] == "exclude"}
    deleted = {row["file_name"] for row in aihub if row["action"] == "delete"}
    assert excluded <= deleted


def test_run_again_keeps_same_result(train_dataset):
    """같은 정제 목록으로 다시 실행하면 결과가 같고, 변경 기록 v3이 생기지 않는다."""
    from pill_detection.data import make_train_dataset

    kwargs = train_dataset["kwargs"]
    aihub_ann = kwargs["aihub_out_dir"] / "train_annotations"
    before = tree_hash(aihub_ann)
    result = make_train_dataset(**kwargs)
    assert result["change_csv"].name == "apply_label_fixes_v2.csv"
    assert not (kwargs["change_csv"].parent / "apply_label_fixes_v3.csv").exists()
    assert tree_hash(aihub_ann) == before
    assert tree_hash(kwargs["kaggle_out_ann_dir"]) == KAGGLE_FIXED_HASH


# ---------- AI Hub 결과 폴더: 노트북 결론 ----------

@pytest.fixture(scope="module")
def aihub_tree(aihub_export):
    """AI Hub 결과의 JSON 전부와, 이미지별로 묶은 annotation."""
    ann_dir, img_dir = aihub_export
    items = read_tree(ann_dir)
    by_image = collections.defaultdict(list)
    for path, data, _ in items:
        by_image[data["images"][0]["file_name"]].append(data)
    return {"ann_dir": ann_dir, "img_dir": img_dir, "items": items, "by_image": by_image}


def test_aihub_passes_strict_kaggle_check(aihub_tree):
    """main의 엄격한 load_annotations를 통과한다. (JSON 하나에 알약 하나, 폴더 = category_id = categories.id, JSON 이름 = 이미지 이름)"""
    assert len(load_annotations(aihub_tree["ann_dir"])) == AIHUB_N_JSON


def test_aihub_key_structure_matches_kaggle(aihub_tree):
    """키 구성과 순서가 대회 train JSON과 같다. (2-8 표 A)"""
    skip_without(KAGGLE_ANN_DIR)
    kaggle = json.loads(next(KAGGLE_ANN_DIR.rglob("*.json")).read_bytes())

    def structure(data):
        return tuple(data), tuple(data["images"][0]), tuple(data["annotations"][0]), tuple(data["categories"][0])

    assert structure(kaggle)[2] == KAGGLE_ANN_KEYS
    assert {structure(data) for _, data, _ in aihub_tree["items"]} == {structure(kaggle)}


def test_aihub_folders_names_and_format(aihub_tree):
    """폴더·이름 규칙과 저장 형식. (2-8 표 A)

    - 경로: <조합>_json/K-<코드>/<이미지 이름>.json
    - K-코드 = category_id = categories.id = drug_N, dl_idx = 코드 숫자, categories.name = dl_name
    - file_name = imgfile, iscrowd·ignore = 0, segmentation = [], 저장 형식은 대회 train JSON과 같다
    """
    for path, data, raw in aihub_tree["items"]:
        image, ann, cat = data["images"][0], data["annotations"][0], data["categories"][0]
        code = ann["category_id"]
        name = image["file_name"]
        assert path.parts == (name.split("_")[0] + "_json", f"K-{code:06d}", name[:-len(".png")] + ".json"), path
        assert image["imgfile"] == name and image["drug_N"] == f"K-{code:06d}" and image["dl_idx"] == str(code), path
        assert cat == {"supercategory": "pill", "id": code, "name": image["dl_name"]}, path
        assert data["type"] == "instances" and (ann["iscrowd"], ann["ignore"], ann["segmentation"]) == (0, 0, []), path
        assert ann["image_id"] == image["id"], path
        assert raw == write_json_bytes(data), path


def test_aihub_ids(aihub_tree):
    """image id는 파일명 순서로 1501부터, annotation id는 5692부터 빈 번호 없이 매긴다. 대회 번호와 겹치지 않는다."""
    image_ids = {}
    for name, anns in aihub_tree["by_image"].items():
        ids = {data["images"][0]["id"] for data in anns}
        assert len(ids) == 1, name
        image_ids[name] = ids.pop()
    assert [image_ids[name] for name in sorted(image_ids)] == list(range(AIHUB_START_IMAGE_ID, AIHUB_START_IMAGE_ID + AIHUB_N_IMAGES))
    ann_ids = sorted(data["annotations"][0]["id"] for _, data, _ in aihub_tree["items"])
    assert ann_ids == list(range(AIHUB_START_ANN_ID, AIHUB_START_ANN_ID + AIHUB_N_JSON))


def test_aihub_counts(aihub_tree):
    """이미지 11,974장, JSON 45,819개, 클래스 116개, 조합 폴더 3,997개. PNG는 JSON의 이미지와 1:1이다. (2-5, 2-7)"""
    by_image = aihub_tree["by_image"]
    assert len(by_image) == AIHUB_N_IMAGES and len(aihub_tree["items"]) == AIHUB_N_JSON
    assert len({data["annotations"][0]["category_id"] for _, data, _ in aihub_tree["items"]}) == AIHUB_N_CLASSES
    assert len({path.parts[0] for path, _, _ in aihub_tree["items"]}) == AIHUB_N_COMBOS
    pngs = {unicodedata.normalize("NFC", p.name) for p in aihub_tree["img_dir"].iterdir()}
    assert pngs == set(by_image) and not any(name.endswith("_index.png") for name in pngs)


def test_aihub_labels_are_clean(aihub_tree):
    """라벨 누락, 클래스 불일치, 좌표 오류가 없다. (2-8 표 B)

    - 이미지마다 클래스 목록 = 파일명의 K-코드 (빠진 알약, 다른 클래스, 남는 box 없음)
    - bbox는 정수 4개, 너비·높이 > 0, 이미지 안. area = 너비 × 높이. 같은 이미지에 똑같은 box 없음
    - 같은 이미지의 JSON끼리 이미지 필드가 같다
    """
    for name, anns in aihub_tree["by_image"].items():
        classes = sorted(data["annotations"][0]["category_id"] for data in anns)
        assert classes == sorted(pill_codes(name)), name
        boxes = []
        for data in anns:
            image, ann = data["images"][0], data["annotations"][0]
            x, y, w, h = ann["bbox"]
            assert all(type(v) is int for v in ann["bbox"]) and w > 0 and h > 0, name
            assert x >= 0 and y >= 0 and x + w <= image["width"] and y + h <= image["height"], name
            assert ann["area"] == w * h, name
            boxes.append(tuple(ann["bbox"]))
        assert len(set(boxes)) == len(boxes), name
        assert len({tuple(data["images"][0][field] for field in IMAGE_FIELDS) for data in anns}) == 1, name


def test_aihub_names_follow_labels(aihub_tree):
    """모든 파일명을 정제 라벨(클래스, 이미지 필드)로 다시 만들면 지금 이름과 같다. (2-6, 2-8 표 C)"""
    for name, anns in aihub_tree["by_image"].items():
        classes = [data["annotations"][0]["category_id"] for data in anns]
        assert expected_file_name(classes, anns[0]["images"][0]) == name


def test_aihub_excluded_and_renamed(aihub_tree):
    """제외한 35장은 없고, 원본과 이름이 다른 이미지는 2-6 이름 표의 13장뿐이다. (2-8 표 C)"""
    skip_without(AIHUB_IMG_DIR)
    raw_photos = {unicodedata.normalize("NFC", p.name) for p in AIHUB_IMG_DIR.glob("*.png") if not p.name.endswith("_index.png")}
    names = set(aihub_tree["by_image"])
    excluded = {row["file_name"] for row in read_csv_rows(LABEL_FIX_CSV) if row["source"] == "aihub" and row["action"] == "exclude"}
    assert len(excluded) == AIHUB_N_EXCLUDED and not excluded & names
    assert names - raw_photos == set(EXPECTED_RENAMES.values())
    assert raw_photos - names == excluded | set(EXPECTED_RENAMES)


def test_aihub_pngs_link_to_raw(aihub_tree):
    """train_images의 PNG는 원본 PNG의 하드링크다. (원본과 다른 디스크면 복사본이라 크기만 본다)"""
    skip_without(AIHUB_IMG_DIR)
    old_name = {new: old for old, new in EXPECTED_RENAMES.items()}
    for name in aihub_tree["by_image"]:
        exported, raw = os.stat(aihub_tree["img_dir"] / name), os.stat(AIHUB_IMG_DIR / old_name.get(name, name))
        if exported.st_dev == raw.st_dev:
            assert exported.st_ino == raw.st_ino, name
        else:
            assert exported.st_size == raw.st_size, name


def test_aihub_matches_notebook_export(aihub_tree):
    """JSON 45,819개가 01 노트북이 내보낸 결과와 바이트까지 같다. 정제 목록의 aihub 줄이 바뀌면 건너뛴다."""
    if fix_rows_sha(LABEL_FIX_CSV, "aihub") != AIHUB_FIX_ROWS_SHA:
        pytest.skip("정제 목록의 aihub 줄이 바뀌어 노트북 결과와 비교할 수 없습니다. "
                    "결과를 확인한 뒤 tests/helpers.py의 AIHUB_EXPORT_HASH와 AIHUB_FIX_ROWS_SHA를 새 값으로 바꾸세요.")
    assert tree_hash(aihub_tree["ann_dir"]) == AIHUB_EXPORT_HASH
