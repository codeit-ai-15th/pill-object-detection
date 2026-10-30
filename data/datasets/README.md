# 2차 실험 목록 CSV

2차 실험(이미지 선택 방법 4가지 비교)의 이미지 목록 `images.csv`와 클래스 표 `classes.csv`입니다.

- [scripts/make_dataset_lists.py](../../scripts/make_dataset_lists.py)가 만들고, 손으로 고치지 않습니다.
- 2차 실험용 val set은 커밋 `58f5d4a`에서 고정했습니다. 지금 입력으로 다시 만들어도 같은지는 아래 명령으로 확인하고, "같음"이 나와야 합니다.
    ```bash
    uv run scripts/make_dataset_lists.py --check
    ```
- 팀원은 이 두 파일대로 학습 폴더를 만들어 학습합니다. 사용법은 [scripts/build_dataset.py](../../scripts/build_dataset.py) 맨 위 docstring을 확인하세요.
    ```bash
    uv run scripts/build_dataset.py --variant v2 --out data/processed/exp2/v2
    ```
- **대회 클래스**는 Kaggle train에 있는 56개와 test에만 나오는 18개를 합친 74개입니다. 채점(val으로 하는 검증)과 제출(Kaggle에 제출하는 csv)은 이 74개 안의 예측만 남기고, 그 밖의 예측은 버립니다.

    | 예측 클래스 | 그 자리의 정답 | 처리 | 점수 영향 |
    | --- | --- | --- | --- |
    | 74개 밖 (AI Hub에만 있는 클래스, `others`) | 74개 밖 알약 | 버림 | 없음 |
    | 74개 밖 | **74개 안 알약** | 버림 | **그 알약은 놓친 것으로 남아 그 클래스 AP가 떨어짐** |
    | 74개 안 | 74개 밖 알약 (IoU 0.5 이상 겹침) | 버림 | 없음 |
    | 74개 안 | 74개 안 알약 | 채점 | 클래스가 맞으면 정답, 틀리면 오답 |
    | 74개 안 | 정답 없음(배경) | 채점 | 오답 |

    - 74개 밖 예측 자체로는 감점이 없습니다. 하지만 대회 클래스 알약을 74개 밖 클래스로 착각하면, 그 알약을 찾지 못한 것과 같아져서 점수가 떨어집니다. 
    - 버리는 것은 그 예측 하나뿐입니다. 같은 알약에 맞는 클래스의 예측이 따로 남아 있으면, 그 예측은 그대로 채점됩니다.
    - 셋째 줄은 val 채점에만 있는 상황입니다. val에는 74개 밖 알약이 737개 섞여 있지만 test에는 하나도 없습니다. 이 알약에 낸 예측을 오답으로 세면 test에서는 일어나지 않는 일로 감점되는 셈입니다. 게다가 이 알약을 학습한 v1~v3만 불리해집니다.
    - Kaggle 제출에서는 CSV에서 그 예측을 뺍니다. test 알약은 모두 74개 클래스 안에 있습니다. 그래서 모델이 74개 밖 클래스로 예측했으면, 그 예측은 test에서 맞을 수가 없습니다. 결과는 val 채점과 같습니다. test 알약을 74개 밖 클래스로 착각했다면, 그 알약은 놓친 것으로 남아 감점됩니다.

## images.csv

이미지 1장이 1줄이고, 12,206줄(Kaggle 232장 + AI Hub 11,974장)입니다.

### 컬럼별 의미

| 열 | 뜻 | 값 |
| --- | --- | --- |
| `source` | 이미지 출처 | `kaggle`, `aihub` |
| `file_name` | 이미지 파일명. AI Hub는 정제 사본의 이름입니다 | `K-000250-..._70_000_200.png` |
| `combo` | 사진에 든 알약의 K-코드 묶음(조합). 한 조합의 사진은 모두 같은 `split`에 있습니다 | `K-000250-000573-002483-006192` |
| `split` | 2차 실험의 train/val. 4가지 방법이 같습니다 | `train`, `val` |
| `exp1_split` | 1차 실험의 train/val. Kaggle만 있고 AI Hub는 빈칸입니다 | `train`, `val`, 빈칸 |
| `v1` | 1이면 v1이 쓰는 사진입니다. v1은 사진을 전부 씁니다 | `0`, `1` |
| `v2` | 1이면 v2가 쓰는 사진입니다. v2는 Kaggle train 전부에, 객체가 적은 클래스부터 클래스별 200개를 목표로 고른 AI Hub 사진을 더합니다 | `0`, `1` |
| `v3` | 1이면 v3가 쓰는 사진입니다. v3는 대회 클래스가 하나라도 든 사진만 씁니다 | `0`, `1` |
| `v4` | 1이면 v4가 쓰는 사진입니다. v4는 v1과 같은 사진을 쓰고, 대회 클래스 밖 알약을 `others` 한 클래스로 묶습니다 | `0`, `1` |
| `n_objects` | 사진 속 알약 수 | `3`, `4` |
| `n_competition` | 그중 대회 클래스 알약 수 | `0`~`4` |
| `n_noncompetition` | 그중 대회 클래스 밖 알약 수 | `0`~`4` |

- 방법 vN의 train은 `vN`이 1이고 `split`이 `train`인 행입니다. val은 `split`이 `val`인 행입니다.
- `v1`과 `v4`는 모든 줄이 1입니다. 두 방법은 사진이 같고 라벨만 다릅니다.
- val 줄은 `v1`~`v4`가 모두 1입니다. 4가지 방법이 같은 val로 채점하기 때문입니다.
- v2의 200개는 목표입니다. 학습 데이터 전체를 기준으로도 200개가 없는 클래스는 20개가 있었습니다. 이 클래스들은 있는 만큼 모두 넣습니다. (예: K-003483은 27개)
- 방법별 train은 v1 11,206장, v2 7,129장, v3 8,974장, v4 11,206장입니다.

### 행 별로 나올 수 있는 경우의 수

`v1~v4`는 네 열의 값을 이어 쓴 것입니다. 예를 들어 `1011`은 v2만 0입니다.

| source | split | exp1_split | v1~v4 | 장수 | 설명 |
| --- | --- | --- | --- | ---: | --- |
| kaggle | train | train | 1111 | 185 | 1차에서 train으로 쓴 Kaggle 사진이고, 2차에서도 네 방법 모두 train에 넣습니다. |
| kaggle | val | val | 1111 | 47 | 1차 val이던 Kaggle 사진입니다. 2차에서도 검증에만 쓰고, 1차와 비교하는 `kaggle` 점수를 이 47장으로 냅니다. |
| aihub | val | 빈칸 | 1111 | 953 | test 클래스 빈도를 닮게 고른 AI Hub 조합의 사진이고, 네 방법 모두 검증에만 씁니다. |
| aihub | train | 빈칸 | 1111 | 5,704 | 대회 클래스가 들어 있고 v2도 고른 사진이라 네 방법 모두 train에 넣습니다. |
| aihub | train | 빈칸 | 1011 | 3,085 | 대회 클래스가 들어 있지만 v2가 클래스별 200개를 채울 때 뽑지 않아 v2에서만 뺍니다. |
| aihub | train | 빈칸 | 1101 | 1,240 | 대회 클래스 밖 알약만 든 사진이라 v3에서 빠집니다. v2는 그 클래스를 채우려고 골랐고, v4에서는 알약이 모두 `others`입니다. |
| aihub | train | 빈칸 | 1001 | 992 | 대회 클래스 밖 알약만 들어 있고 v2도 고르지 않아 v1과 v4의 train에만 넣습니다. |

- 합계는 12,206장이고, 그중 val은 1,000장(Kaggle 47장 + AI Hub 953장)입니다.
- val의 AI Hub 953장 중 737장에는 대회 클래스 밖 알약이 1개씩 섞여 있습니다. 채점할 때 이 알약에 낸 예측은 무시합니다. (규칙: [evaluate.py](../../src/pill_detection/evaluate.py) 모듈 docstring)

**예시 행 (위 표의 순서대로 하나씩)**

```csv
source,file_name,combo,split,exp1_split,v1,v2,v3,v4,n_objects,n_competition,n_noncompetition
kaggle,K-001900-016548-019607-029451_0_2_0_2_70_000_200.png,K-001900-016548-019607-029451,train,train,1,1,1,1,4,4,0
kaggle,K-003351-003832-036637_0_2_0_2_90_000_200.png,K-003351-003832-036637,val,val,1,1,1,1,3,3,0
aihub,K-000250-002483-005886-012778_0_2_0_2_70_000_200.png,K-000250-002483-005886-012778,val,,1,1,1,1,4,3,1
aihub,K-000250-000573-002483-006192_0_2_0_2_70_000_200.png,K-000250-000573-002483-006192,train,,1,1,1,1,4,2,2
aihub,K-000250-000573-005002-006192_0_2_0_2_70_000_200.png,K-000250-000573-005002-006192,train,,1,0,1,1,4,1,3
aihub,K-006697-006835-023319-030850_0_2_0_2_70_000_200.png,K-006697-006835-023319-030850,train,,1,1,0,1,4,0,4
aihub,K-006697-006835-020004-023319_0_2_0_2_70_000_200.png,K-006697-006835-020004-023319,train,,1,0,0,1,4,0,4
```

## classes.csv

클래스 1개가 1줄이고, 맨 끝에 `others` 1줄이 있습니다. 118줄 + `others` 1줄 = 119줄입니다.

### 컬럼별 의미

| 열 | 뜻 | 값 |
| --- | --- | --- |
| `category_id` | K-코드 숫자. `others`는 `1000000`입니다 | `3351` |
| `k_code` | K-코드 | `K-003351`, `others` |
| `name` | 제품명 | `일양하이트린정 2mg` |
| `group` | 클래스 그룹. `kaggle`은 Kaggle train에 있음, `test_only`는 Kaggle train에는 없고 test에 나옴, `aihub_only`는 AI Hub에만 있음 | `kaggle`, `test_only`, `aihub_only`, `others` |
| `competition` | 1이면 대회 클래스(채점, 제출 대상)입니다 | `0`, `1` |
| `idx_v1`~`idx_v4` | 그 방법의 모델 클래스 번호(0부터). 빈칸이면 그 방법이 학습하지 않는 클래스입니다 | `0`~`117`, 빈칸 |
| `n_test` | test 842장에 등장하는 해당 클래스의 객체 수 추정치(노트북 02 검수표). `others`는 빈칸입니다 | `369` |
| `n_val` | 2차 실험 val에 등장하는 해당 클래스의 객체 수. Kaggle과 AI Hub의 모든 객체를 셉니다 | `18` |
| `n_train_v1`~`n_train_v4` | 그 방법 train의 객체 수. 학습 라벨 기준이라, v4에서 대회 클래스 밖 클래스는 0이고 `others` 줄에 합쳐집니다 | `27` |

- `idx` 번호는 방법(v1, v2, v3, v4)마다 다릅니다. (예: K-027653은 v1: 78, v2: 78, v3: 65, v4: 49)
    - 학습 폴더의 YOLO `data.yaml` names 순서, COCO categories의 `idx`와 같습니다.
- 채점과 제출은 번호가 아니라 K-코드(`category_id`)로 합니다. 모델이 낸 번호는 K-코드로 바꿔서 넘깁니다. (예측 형식: [evaluate.py](../../src/pill_detection/evaluate.py) 모듈 docstring)

### 행 별로 나올 수 있는 경우의 수

| group | competition | idx_v{n} | 줄 수 | 설명 |
| --- | --- | --- | ---: | --- |
| kaggle | 1 | v1~v4 | 56 | Kaggle train에 있는 대회 클래스라 네 방법 모두 학습하고, 채점과 제출 대상입니다. |
| test_only | 1 | v1~v4 | 18 | Kaggle train에는 없지만 test에 나오는 클래스입니다. AI Hub 사진으로만 배우고, 채점과 제출 대상입니다. |
| aihub_only | 0 | v1~v3 | 22 | AI Hub에만 있는 채점 밖 클래스입니다. 대회 클래스와 한 사진에 찍힌 적이 있어 v3까지 학습하고, v4에서는 `others`로 합칩니다. |
| aihub_only | 0 | v1, v2 | 22 | AI Hub에만 있는 채점 밖 클래스입니다. 대회 클래스와 한 사진에 찍힌 적이 없어 v1, v2에서만 학습하고, val에도 나오지 않습니다. v4에서는 `others`로 합칩니다. |
| others | 0 | v4 | 1 | v4에서만 쓰는 묶음 클래스입니다. 대회 클래스 밖 44종을 하나로 합친 것이고, 채점과 제출은 하지 않습니다. |

**예시 행 (위 표의 순서대로 하나씩)**

```csv
category_id,k_code,name,group,competition,idx_v1,idx_v2,idx_v3,idx_v4,n_test,n_val,n_train_v1,n_train_v2,n_train_v3,n_train_v4
3351,K-003351,일양하이트린정 2mg,kaggle,1,5,5,5,2,139,29,128,128,128,128
27653,K-027653,세비카정 10/40mg,test_only,1,78,78,65,49,73,77,390,200,390,390
250,K-000250,마그밀정(수산화마그네슘),aihub_only,0,0,0,0,,0,18,521,342,521,0
20753,K-020753,인사돌플러스정 35mg/PTP,aihub_only,0,56,56,,,0,0,90,90,0,0
1000000,others,others,others,0,,,,74,,737,,,,16552
```

### 눈여겨볼 행

- K-003483: test에 가장 많이 나오지만(369개) AI Hub에 없어서, val은 1차 val Kaggle 사진의 18개, train은 네 방법 모두 27개뿐입니다.
- K-001900: test 비율로는 val 목표가 128개지만 상한(그 클래스 AI Hub 객체의 20%) 때문에 32개로 줄였습니다. 알약 조합 별로 train/val을 나누므로, 10조합을 val에 넣어 상한을 넘기지 않도록 30개로 채웠습니다.
- `others`: `n_val` 737과 `n_train_v4` 16,552는 `aihub_only` 44종의 합입니다.

## 쓰는 법

```python
import pandas as pd

images = pd.read_csv("data/datasets/images.csv")
v2 = images[images.v2 == 1]
train, val = v2[v2.split == "train"], v2[v2.split == "val"]  # 7,129장, 1,000장

classes = pd.read_csv("data/datasets/classes.csv")
competition = classes[classes.competition == 1]  # 채점, 제출하는 74개
```

- 학습할 때는 CSV를 직접 읽지 않고 `build_dataset.py`가 만든 학습 폴더를 씁니다.
- val을 고른 규칙은 [datasets.py](../../src/pill_detection/datasets.py)의 `select_val` docstring을, 방법별 train 규칙은 `select_variants` docstring을 봅니다.
