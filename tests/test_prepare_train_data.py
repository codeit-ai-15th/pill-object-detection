"""AI Hub 받기·풀기 스크립트(scripts/)의 폴더 규칙과 건너뛰기 판단을 가짜 zip으로 본다. 네트워크는 쓰지 않는다."""

import sys
import zipfile

import pytest

from helpers import PROJECT_ROOT, read_csv_rows

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import download_aihub  # noqa: E402
import extract_aihub  # noqa: E402
import prepare_train_data  # noqa: E402

from pill_detection.data import check_aihub_file  # noqa: E402

LABEL_JSON = "K-000001-000002_json/K-000001/K-000001-000002_0_2_0_2_70_000_200.json"
PHOTO = "K-000001-000002_0_2_0_2_70_000_200.png"


def make_zip(path, members):
    """members(zip 안 이름 → 내용)로 zip을 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_download_goes_to_raw_folders(tmp_path):
    """zip은 data/external/aihub/raw/의 train_annotations(라벨)나 train_images(이미지)에 받는다."""
    assert download_aihub.DEFAULT_OUT == PROJECT_ROOT / "data" / "external" / "aihub" / "raw"
    assert download_aihub.target_path(tmp_path, "TL_1_조합.zip") == tmp_path / "train_annotations" / "TL_1_조합.zip"
    assert download_aihub.target_path(tmp_path, "VS_1_조합.zip") == tmp_path / "train_images" / "VS_1_조합.zip"


def test_extract_label_zip_in_place(tmp_path):
    """라벨 zip은 zip이 있는 폴더에 zip 안 구조 그대로 풀고, 다 풀면 zip을 지우고, 나온 파일을 sources.csv에 적는다."""
    raw, sources = tmp_path / "raw", tmp_path / "sources.csv"
    zip_path = make_zip(raw / "train_annotations" / "TL_9_조합.zip",
                        {LABEL_JSON: "{}", "__MACOSX/._a.json": "", "K-000001-000002_json/.DS_Store": ""})
    written, skipped, ignored = extract_aihub.extract_zip(zip_path, raw, sources)
    assert (written, skipped, ignored) == (1, 0, 2)
    assert (raw / "train_annotations" / LABEL_JSON).read_text() == "{}"
    assert not zip_path.exists()
    assert read_csv_rows(sources) == [{"zip": "TL_9_조합.zip", "path": f"train_annotations/{LABEL_JSON}"}]


def test_extract_image_zip_flat_and_keep_zip(tmp_path):
    """이미지 zip은 하위 폴더 없이 한 폴더에 푼다. keep_zip=True면 zip을 남긴다. sources.csv의 다른 zip 줄은 그대로 둔다."""
    raw, sources = tmp_path / "raw", tmp_path / "sources.csv"
    sources.write_text("zip,path\nTL_9_조합.zip,train_annotations/a.json\n", encoding="utf-8")
    zip_path = make_zip(raw / "train_images" / "TS_9_조합.zip",
                        {f"TS_9/x/{PHOTO}": b"png", "TS_9/x/K-000001-000002_index.png": b"idx", "TS_9/readme.txt": b""})
    written, _, ignored = extract_aihub.extract_zip(zip_path, raw, sources, keep_zip=True)
    assert (written, ignored) == (2, 1)
    assert (raw / "train_images" / PHOTO).read_bytes() == b"png"
    assert zip_path.exists()
    assert read_csv_rows(sources) == [
        {"zip": "TL_9_조합.zip", "path": "train_annotations/a.json"},
        {"zip": "TS_9_조합.zip", "path": "train_images/K-000001-000002_0_2_0_2_70_000_200.png"},
        {"zip": "TS_9_조합.zip", "path": "train_images/K-000001-000002_index.png"},
    ]


def test_extract_rejects_banned_zip(tmp_path):
    """대회에서 금지한 zip은 풀지 않는다."""
    raw = tmp_path / "raw"
    zip_path = make_zip(raw / "train_annotations" / "TL_2_조합.zip", {LABEL_JSON: "{}"})
    with pytest.raises(ValueError):
        extract_aihub.extract_zip(zip_path, raw, tmp_path / "sources.csv")
    assert not (raw / "train_annotations" / LABEL_JSON).exists()


def test_zip_list():
    """받을 zip은 허용 조합 라벨 8개와 이미지 8개다. 금지 파일은 없다."""
    names = [name for _, name in prepare_train_data.AIHUB_ZIPS]
    assert len(names) == len(set(names)) == 16
    assert sorted(name for name in names if name[1] == "L") == \
        ["TL_1_조합.zip", "TL_3_조합.zip", "TL_4_조합.zip", "TL_5_조합.zip", "TL_6_조합.zip", "TL_7_조합.zip", "TL_8_조합.zip", "VL_1_조합.zip"]
    assert sorted(name for name in names if name[1] == "S") == \
        ["TS_1_조합.zip", "TS_3_조합.zip", "TS_4_조합.zip", "TS_5_조합.zip", "TS_6_조합.zip", "TS_7_조합.zip", "TS_8_조합.zip", "VS_1_조합.zip"]
    for name in names:
        check_aihub_file(name)


def test_zips_to_fetch(tmp_path):
    """sources.csv에 있고 그 파일이 모두 있는 zip만 건너뛴다."""
    raw, sources = tmp_path / "raw", tmp_path / "sources.csv"
    (raw / "train_annotations").mkdir(parents=True)
    (raw / "train_annotations" / "a.json").write_text("{}")
    sources.write_text("zip,path\nTL_9_조합.zip,train_annotations/a.json\nTS_9_조합.zip,train_images/b.png\n", encoding="utf-8")
    zips = [(1, "TL_9_조합.zip"), (2, "TS_9_조합.zip"), (3, "TS_8_조합.zip")]
    assert prepare_train_data.zips_to_fetch(zips, raw, sources) == [(2, "TS_9_조합.zip"), (3, "TS_8_조합.zip")]
    assert prepare_train_data.zips_to_fetch(zips, raw, tmp_path / "none.csv") == zips
