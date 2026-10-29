"""Kaggle 원본 라벨 수정(make_fixed_annotations)이 data.py를 고치기 전과 같게 동작하는지 본다.

기준은 main(e9100f9)의 data.py다. 이 파일의 테스트는 data.py를 고치기 전에 먼저 모두 통과시켰다.
결과 파일과 오류만 보고, 함수가 돌려주는 dict의 키 이름은 보지 않는다. (합친 정제 목록에서 키 이름이 바뀌기 때문)
"""

import collections
import json
from pathlib import Path

import pytest

from conftest import skip_without
from helpers import (
    CHANGE_CSV_V1,
    KAGGLE_ANN_DIR,
    KAGGLE_FIXED_HASH,
    KAGGLE_IMG_DIR,
    LEGACY_FIX_CSV,
    read_csv_rows,
    tree_hash,
    write_json_bytes,
)
from pill_detection.data import IMAGE_FIELDS, make_fixed_annotations, read_label_fixes, write_label_changes

LEGACY_HEADER = "file_name,category_id,action,bbox,area\n"
FN = "K-003351-016262-018357_0_2_0_2_75_000_200.png"
GOOD_ROW = f'{FN},18357,modify,"[567, 625, 311, 315]",97965\n'


# ---------- 가짜 데이터로 하는 검사 (실제 데이터 없이 돈다) ----------

@pytest.mark.parametrize("text, message", [
    ("file_name,category_id,action,bbox\n" + f'{FN},18357,modify,"[567, 625, 311, 315]"\n', "열 이름"),
    (LEGACY_HEADER + f"{FN},18357,modify,,97965\n", "빈칸"),
    (LEGACY_HEADER + f'{FN},18357,delete,"[567, 625, 311, 315]",97965\n', "action"),
    (LEGACY_HEADER + f'{FN},abc,modify,"[567, 625, 311, 315]",97965\n', "숫자"),
    (LEGACY_HEADER + f'{FN},18357,modify,"[567, 625, 311]",97965\n', "bbox"),
    (LEGACY_HEADER + f'{FN},18357,modify,"[567, 625, 311, 315]",1\n', "area"),
    (LEGACY_HEADER + GOOD_ROW + GOOD_ROW, "같습니다"),
])
def test_read_label_fixes_rejects_bad_legacy_csv(tmp_path, text, message):
    """옛 5열 정제 목록의 형식 오류는 ValueError로 멈춘다."""
    path = tmp_path / "label_fixes.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_label_fixes(path)


def test_read_label_fixes_reads_legacy_csv():
    """main의 옛 정제 목록을 읽는다. (add 8줄, modify 13줄)"""
    fixes = read_label_fixes(LEGACY_FIX_CSV)
    assert collections.Counter(fix["action"] for fix in fixes) == {"add": 8, "modify": 13}


def _record(file_name, category_id, bbox, area):
    """load_annotations()가 돌려주는 값 하나와 같은 모양의 가짜 기록."""
    return {"json_path": f"{file_name.split('_')[0]}_json/K-{category_id:06d}/{Path(file_name).stem}.json",
            "file_name": file_name, "image_id": 1, "category_id": category_id, "ann_id": category_id,
            "bbox": bbox, "area": area}


def test_write_label_changes_versions(tmp_path):
    """내용이 같으면 새 파일을 만들지 않고, 다르면 _v2로 저장한다. CRLF와 BOM 차이는 같은 내용으로 본다."""
    base = {("a_0.png", 1): _record("a_0.png", 1, [0, 0, 2, 2], 4)}
    out = {("a_0.png", 1): _record("a_0.png", 1, [0, 0, 3, 2], 6), ("a_0.png", 2): _record("a_0.png", 2, [5, 5, 1, 1], 1)}
    csv_path = tmp_path / "apply_label_fixes.csv"

    path, changes = write_label_changes(base, out, csv_path)
    assert path == csv_path
    assert [change["action"] for change in changes] == ["add", "modify"]
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == \
        "file_name,category_id,action,bbox_before,bbox_after,area_before,area_after"

    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_bytes(("﻿" + text.replace("\n", "\r\n")).encode("utf-8"))
    path, _ = write_label_changes(base, out, csv_path)
    assert path == csv_path and not (tmp_path / "apply_label_fixes_v2.csv").exists()

    del out[("a_0.png", 2)]
    path, changes = write_label_changes(base, out, csv_path)
    assert path.name == "apply_label_fixes_v2.csv"
    assert [change["action"] for change in changes] == ["modify"]


# ---------- 실제 데이터로 하는 검사 (data/raw 필요) ----------

@pytest.fixture(scope="module")
def kaggle_fixed(tmp_path_factory):
    """옛 정제 목록으로 make_fixed_annotations를 임시 폴더에 실행한 결과."""
    skip_without(KAGGLE_ANN_DIR, KAGGLE_IMG_DIR)
    root = tmp_path_factory.mktemp("kaggle")
    out_dir = root / "train_annotations"
    result = make_fixed_annotations(LEGACY_FIX_CSV, KAGGLE_ANN_DIR, KAGGLE_IMG_DIR, out_dir, root / "apply_label_fixes.csv")
    return result, out_dir


@pytest.fixture(scope="module")
def raw_index():
    """Kaggle 원본 JSON: ann_dir 기준 경로(/) → (JSON, 바이트)."""
    skip_without(KAGGLE_ANN_DIR)
    return {p.relative_to(KAGGLE_ANN_DIR).as_posix(): (json.loads(p.read_bytes()), p.read_bytes())
            for p in sorted(KAGGLE_ANN_DIR.rglob("*.json"))}


def _json_path(file_name, category_id):
    return f"{file_name.split('_')[0]}_json/K-{category_id:06d}/{Path(file_name).stem}.json"


def test_counts_and_change_log(kaggle_fixed):
    """JSON 763개 → 771개이고, 변경 기록은 git에 있는 apply_label_fixes.csv와 같다."""
    result, _ = kaggle_fixed
    assert (result["n_base"], result["n_out"]) == (763, 771)
    assert result["change_csv"].read_text(encoding="utf-8") == CHANGE_CSV_V1.read_text(encoding="utf-8")


def test_untouched_json_are_byte_identical(kaggle_fixed, raw_index):
    """정제 목록에 없는 JSON은 원본과 바이트까지 같다."""
    _, out_dir = kaggle_fixed
    fixed = {_json_path(row["file_name"], int(row["category_id"])) for row in read_csv_rows(LEGACY_FIX_CSV)}
    untouched = [path for path in raw_index if path not in fixed]
    assert len(untouched) == 763 - 13
    for path in untouched:
        assert (out_dir / path).read_bytes() == raw_index[path][1], path


def test_modify_changes_only_bbox_and_area(kaggle_fixed, raw_index):
    """modify 줄의 JSON은 원본에서 bbox와 area만 정제 목록 값으로 바뀐다. 저장 형식도 원본과 같다."""
    _, out_dir = kaggle_fixed
    rows = [row for row in read_csv_rows(LEGACY_FIX_CSV) if row["action"] == "modify"]
    assert len(rows) == 13
    for row in rows:
        path = _json_path(row["file_name"], int(row["category_id"]))
        expected = json.loads(raw_index[path][1])
        expected["annotations"][0]["bbox"] = json.loads(row["bbox"])
        expected["annotations"][0]["area"] = int(row["area"])
        assert (out_dir / path).read_bytes() == write_json_bytes(expected), path


def test_add_json_takes_drug_and_image_info(kaggle_fixed, raw_index):
    """add 줄의 새 JSON은 알약 정보 JSON을 바탕으로, 이미지 필드 12개만 같은 이미지의 JSON 값으로 바꾼다.

    알약 정보 JSON: 같은 조합의 같은 약 폴더에서 경로 순서로 첫 JSON
    이미지 정보 JSON: 같은 이미지의 JSON 중 경로 순서로 첫 JSON
    새 annotation id는 원본의 가장 큰 id 다음부터 줄 순서로 매긴다.
    """
    _, out_dir = kaggle_fixed
    rows = [row for row in read_csv_rows(LEGACY_FIX_CSV) if row["action"] == "add"]
    assert len(rows) == 8
    next_id = max(data["annotations"][0]["id"] for data, _ in raw_index.values()) + 1
    for row in rows:
        fn, code = row["file_name"], int(row["category_id"])
        drug_dir = f"{fn.split('_')[0]}_json/K-{code:06d}/"
        drug_path = min(path for path in raw_index if path.startswith(drug_dir))
        image_path = min(path for path, (data, _) in raw_index.items() if data["images"][0]["file_name"] == fn)
        expected = json.loads(raw_index[drug_path][1])
        image = raw_index[image_path][0]["images"][0]
        for field in IMAGE_FIELDS:
            expected["images"][0][field] = image[field]
        ann = expected["annotations"][0]
        ann["id"], ann["image_id"] = next_id, image["id"]
        ann["bbox"], ann["area"] = json.loads(row["bbox"]), int(row["area"])
        next_id += 1
        assert (out_dir / _json_path(fn, code)).read_bytes() == write_json_bytes(expected), fn


def test_output_hash_matches_main(kaggle_fixed):
    """사본 JSON 771개가 main 코드의 결과와 바이트까지 같다."""
    _, out_dir = kaggle_fixed
    assert tree_hash(out_dir) == KAGGLE_FIXED_HASH
