"""Drive 업로드용 zip 스크립트(scripts/pack_drive_data.py)를 가짜 데이터 폴더로 본다."""

import sys
import zipfile

from helpers import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import pack_drive_data  # noqa: E402

PACKS = {
    "kaggle": [("data/raw/train_images", ".png", 2), ("data/processed/train_annotations", ".json", 1)],
    "test_images": [("data/raw/test_images", ".png", 1)],
}


def make_repo(root):
    """PACKS 기대값에 맞는 가짜 레포를 만든다. macOS 숨김 파일과 다른 확장자 파일도 섞어 둔다."""
    files = {
        "data/raw/train_images/1.png": b"png1",
        "data/raw/train_images/2.png": b"png2",
        "data/raw/train_images/.DS_Store": b"",
        "data/raw/train_images/._1.png": b"",
        "data/processed/train_annotations/K-000001_json/K-000001/a.json": b'{"a": 1}' * 100,
        "data/processed/train_annotations/readme.txt": b"",
        "data/raw/test_images/9.png": b"png9",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def test_pack_writes_posix_paths_and_compression(tmp_path):
    """zip 안 경로는 data/...로 시작하고 구분자가 /이다. PNG는 무압축, JSON은 압축이다. 숨김 파일은 빠진다."""
    root, out = make_repo(tmp_path / "repo"), tmp_path / "drive"
    assert pack_drive_data.main(["--out", str(out)], root=root, packs=PACKS) == 0
    with zipfile.ZipFile(out / "kaggle.zip") as zf:
        infos = {info.filename: info for info in zf.infolist()}
        assert list(infos) == [  # PACKS의 폴더 순서, 폴더 안에서는 경로 순서
            "data/raw/train_images/1.png",
            "data/raw/train_images/2.png",
            "data/processed/train_annotations/K-000001_json/K-000001/a.json",
        ]
        assert infos["data/raw/train_images/1.png"].compress_type == zipfile.ZIP_STORED
        assert infos["data/processed/train_annotations/K-000001_json/K-000001/a.json"].compress_type == zipfile.ZIP_DEFLATED
        assert zf.read("data/raw/train_images/2.png") == b"png2"
    with zipfile.ZipFile(out / "test_images.zip") as zf:
        assert zf.namelist() == ["data/raw/test_images/9.png"]
    assert not list(out.glob("*.part"))


def test_pack_refuses_wrong_count(tmp_path, capsys):
    """파일 수가 기대값과 다르면 zip을 하나도 만들지 않고, 다시 만드는 명령을 알려 준다."""
    root, out = make_repo(tmp_path / "repo"), tmp_path / "drive"
    (root / "data/raw/test_images/10.png").write_bytes(b"extra")
    assert pack_drive_data.main(["--out", str(out)], root=root, packs=PACKS) == 1
    assert "data/raw/test_images의 .png 파일이 2개입니다 (기대값 1개)" in capsys.readouterr().out
    assert not out.exists()


def test_pack_refuses_out_inside_repo(tmp_path):
    """zip을 레포 안에 쓰지 않는다."""
    root = make_repo(tmp_path / "repo")
    assert pack_drive_data.main(["--out", str(root / "data" / "zips")], root=root, packs=PACKS) == 1
    assert not (root / "data" / "zips").exists()


def test_pack_only_and_existing_zip(tmp_path):
    """--only는 그 zip만 만든다. 이미 있는 zip은 덮어쓰지 않는다."""
    root, out = make_repo(tmp_path / "repo"), tmp_path / "drive"
    assert pack_drive_data.main(["--out", str(out), "--only", "test_images"], root=root, packs=PACKS) == 0
    assert sorted(p.name for p in out.iterdir()) == ["test_images.zip"]
    assert pack_drive_data.main(["--out", str(out), "--only", "test_images"], root=root, packs=PACKS) == 1
    assert pack_drive_data.main(["--out", str(out), "--only", "kaggle"], root=root, packs=PACKS) == 0


def test_packs_match_readme_counts():
    """기대 파일 수는 학습 데이터 준비 결과(Kaggle 사본 771, AI Hub 11,974장과 45,819개)와 같다."""
    from helpers import AIHUB_N_IMAGES, AIHUB_N_JSON, KAGGLE_FIXED_HASH

    counts = {folder: n for packs in pack_drive_data.PACKS.values() for folder, _, n in packs}
    assert counts["data/processed/train_annotations"] == KAGGLE_FIXED_HASH[0]
    assert counts["data/processed/ai_hub/train_images"] == AIHUB_N_IMAGES
    assert counts["data/processed/ai_hub/train_annotations"] == AIHUB_N_JSON
