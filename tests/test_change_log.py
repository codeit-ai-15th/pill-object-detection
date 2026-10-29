"""변경 기록 묶음(apply_label_fixes.csv, _v2, _v3 …)의 규칙을 가짜 데이터로 본다. 실제 데이터 없이 돈다.

- 어느 함수로 만들든 같은 묶음에 쌓인다. _vN 파일 경로를 넣어도 그 묶음의 다음 번호로 저장한다.
- 최신 기록이 Kaggle + AI Hub 형식(TRAIN_CHANGE_COLUMNS)이면, Kaggle만 다루는 옛 호출도 그 형식으로 쓰고 aihub 줄은 이어받는다.
- 줄 순서만 다른 기록은 같은 내용으로 본다.
"""

from pathlib import Path

from helpers import CHANGE_CSV_V1, read_csv_rows
from pill_detection.data import TRAIN_CHANGE_COLUMNS, write_label_changes

CHANGE_CSV_V2 = CHANGE_CSV_V1.with_name("apply_label_fixes_v2.csv")  # git에 있는 Kaggle + AI Hub 기록

KAGGLE_FN = "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png"
AIHUB_FN = "K-003614-016262-027777-028763_0_2_0_2_70_000_200.png"


def _record(file_name, category_id, bbox):
    """load_annotations()가 돌려주는 값 하나와 같은 모양의 가짜 기록."""
    return {"json_path": f"{file_name.split('_')[0]}_json/K-{category_id:06d}/{Path(file_name).stem}.json",
            "file_name": file_name, "image_id": 1, "category_id": category_id, "ann_id": category_id,
            "bbox": bbox, "area": bbox[2] * bbox[3]}


def _kaggle(bbox):
    """Kaggle 알약 하나의 기준과 결과. 결과 bbox만 바꿔 쓴다."""
    key = (KAGGLE_FN, 16548)
    return {key: _record(*key, [88, 255, 366, 209])}, {key: _record(*key, bbox)}


def _train_series(tmp_path):
    """make_train_dataset처럼 Kaggle과 AI Hub를 함께 기록한 묶음을 만든다. (Kaggle만 담은 7열 1번 파일 + 10열 _v2)

    Returns:
        Path: 묶음의 1번 파일 경로.
    """
    csv_path = tmp_path / "apply_label_fixes.csv"
    kaggle_base, kaggle_out = _kaggle([99, 862, 245, 239])
    write_label_changes(kaggle_base, kaggle_out, csv_path)

    old_key, new_key = ("K-003614-027777-028763-030308_0_2_0_2_70_000_200.png", 30308), (AIHUB_FN, 16262)
    added = ("K-003614-027777-028763-030308_0_2_0_2_75_000_200.png", 3614)
    base = {**kaggle_base, old_key: _record(*old_key, [10, 10, 5, 5])}
    out = {**kaggle_out, new_key: _record(*new_key, [12, 10, 5, 5]), added: _record(*added, [1, 2, 3, 4])}
    sources = {key: "kaggle" for key in kaggle_base} | {key: "aihub" for key in (old_key, new_key, added)}
    path, _ = write_label_changes(base, out, csv_path, key_map={old_key: new_key}, sources=sources)
    assert path.name == "apply_label_fixes_v2.csv"
    return csv_path


def test_numbered_path_joins_same_series(tmp_path):
    """_v2 파일 경로를 넣어도 같은 묶음의 다음 번호(_v3)로 저장한다. _v2_v2를 만들지 않는다."""
    csv_path = _train_series(tmp_path)
    kaggle_base, kaggle_out = _kaggle([100, 862, 245, 239])
    path, _ = write_label_changes(kaggle_base, kaggle_out, csv_path.with_name("apply_label_fixes_v2.csv"))
    assert path.name == "apply_label_fixes_v3.csv"
    assert not (tmp_path / "apply_label_fixes_v2_v2.csv").exists()


def test_row_order_does_not_matter(tmp_path):
    """최신 기록과 줄 순서만 다르면 같은 내용으로 보고 새 파일을 만들지 않는다."""
    csv_path = _train_series(tmp_path)
    latest = tmp_path / "apply_label_fixes_v2.csv"
    header, *rows = latest.read_text(encoding="utf-8").splitlines()
    latest.write_text("\n".join([header, *reversed(rows)]) + "\n", encoding="utf-8")

    kaggle_base, kaggle_out = _kaggle([99, 862, 245, 239])
    path, _ = write_label_changes(kaggle_base, kaggle_out, csv_path)
    assert path == latest
    assert not (tmp_path / "apply_label_fixes_v3.csv").exists()


def test_kaggle_only_call_keeps_aihub_rows_when_unchanged(tmp_path):
    """최신 기록이 Kaggle + AI Hub 형식이고 Kaggle 결과가 그대로면, Kaggle만 다루는 호출은 새 파일을 만들지 않는다."""
    csv_path = _train_series(tmp_path)
    kaggle_base, kaggle_out = _kaggle([99, 862, 245, 239])
    path, changes = write_label_changes(kaggle_base, kaggle_out, csv_path)
    assert path.name == "apply_label_fixes_v2.csv"
    assert not (tmp_path / "apply_label_fixes_v3.csv").exists()
    # 돌려준 변경 목록은 기록 파일 내용과 같다 (이어받은 aihub 줄 포함)
    assert sorted((c["source"], c["file_name"], c["action"]) for c in changes) == \
        sorted((r["source"], r["file_name"], r["action"]) for r in read_csv_rows(path))


def test_kaggle_only_call_carries_aihub_rows_when_changed(tmp_path):
    """Kaggle 결과가 바뀌면 다음 번호에 같은 형식으로 저장하고, aihub 줄은 최신 기록의 글자 그대로 이어받는다."""
    csv_path = _train_series(tmp_path)
    kaggle_base, kaggle_out = _kaggle([100, 862, 245, 239])
    path, _ = write_label_changes(kaggle_base, kaggle_out, csv_path)
    assert path.name == "apply_label_fixes_v3.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(TRAIN_CHANGE_COLUMNS)

    old, new = read_csv_rows(tmp_path / "apply_label_fixes_v2.csv"), read_csv_rows(path)
    assert [r for r in new if r["source"] == "aihub"] == [r for r in old if r["source"] == "aihub"]
    kaggle = [r for r in new if r["source"] == "kaggle"]
    assert [(r["file_name"], r["file_name_after"], r["category_id_after"], r["bbox_before"], r["bbox_after"])
            for r in kaggle] == [(KAGGLE_FN, KAGGLE_FN, "16548", "[88, 255, 366, 209]", "[100, 862, 245, 239]")]


def test_carried_rows_keep_text_of_real_v2(tmp_path):
    """git에 있는 v2의 aihub 줄을 이어받으면 글자까지 같다. (bbox가 []이고 area에 리스트가 든 줄도 있다)"""
    csv_path = tmp_path / "apply_label_fixes.csv"
    (tmp_path / CHANGE_CSV_V2.name).write_bytes(CHANGE_CSV_V2.read_bytes())
    path, _ = write_label_changes({}, {}, csv_path)  # Kaggle 변경이 없는 기록 → aihub 줄만 남은 v3
    assert path.name == "apply_label_fixes_v3.csv"

    def aihub_lines(p):
        return sorted(line for line in p.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
                      if line.startswith("aihub,"))
    assert aihub_lines(path) == aihub_lines(CHANGE_CSV_V2)
    assert len(aihub_lines(path)) == 1199
