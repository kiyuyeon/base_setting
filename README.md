# cuda_test

**나이 추정 모델 실험장** (C3AE / SSR-Net / YOLO). 이름과 달리 CUDA 테스트는 `cuda_test.ipynb` 하나뿐입니다.

## 코드

| 경로 | 역할 |
|---|---|
| `c3ae/c3ae_net.py` | C3AE 모델 정의 |
| `c3ae/data_loader.py` | 데이터 로더 |
| `c3ae/preprocess.py` | 전처리 |
| `c3ae/c3ae.ipynb` | C3AE 학습·평가 |
| `age.ipynb` | 나이 추정 실험 (SSR-Net 등) |
| `test.ipynb` | 테스트 |
| `cuda_test.ipynb` | GPU 동작 확인 |
| `dockerfile` | 실행 환경 |

## 데이터 (총 3.1G)

| 경로 | 크기 | 내용 |
|---|---:|---|
| `yolo_megaage_dataset/` | 2.6G | MegaAge를 YOLO 형식으로 변환 (`data.yaml` + images/labels). 리사이즈된 별개 파일 |
| `All-Age-Faces Dataset/` | 455M | AAF 원본 (aligned faces, key points, image sets) |

### ⚠️ megaage_asian 위치 변경 (2026-08-19)

`megaage_asian/`은 **`../age_data/megaage_asian`으로 일원화**했습니다.
43,952개 파일 전량이 그쪽에 동일하게 존재함을 해시로 확인했고, 그쪽에는 라벨 CSV 3개
(`labels_train.csv`, `labels_test.csv`, `missing_from_list.csv`)까지 있어 상위 집합입니다.

**코드 수정 필요:**
- `c3ae/c3ae_net.py:391`, `c3ae/c3ae.ipynb`, `age.ipynb` 의 `DATA_ROOT`가
  `C:\Users\chwo6\OneDrive\바탕 화면\cuda_test\megaage_asian` 로 하드코딩되어 있습니다
  (이미 존재하지 않는 옛 OneDrive 경로) → `../age_data/megaage_asian` 로 변경
- 파일명 규칙이 다릅니다: 여기 `15474.jpg` → age_data 쪽 `015474.jpg` (6자리 zero-padding).
  `_splits/*.txt`의 이름도 padding 없는 형식이라 로딩 시 변환 필요

## 산출물

- `c3ae/checkpoints/` — 학습 체크포인트
- `preds_ssrnet_aaf.csv` — SSR-Net 예측 결과
- `_splits/` — train/val 분할 목록 (`train_names.txt`, `train_ages.txt` 등)

## 관련 폴더

`../age_detection` (C3AE vs YOLO 최종 비교), `../age_data` (MUFAC 데이터셋)
