# Custom Co-Correcting for Noisy Labels

이 프로젝트는 Co-Correcting 알고리즘을 기반으로 커스텀 데이터셋(예: PetSkin)을 학습할 수 있도록 확장한 버전입니다.  
ASAM Optimizer와 ANL (Adaptive Negative Learning)을 적용하여 라벨 노이즈 환경에서도 안정적인 일반화 성능을 제공합니다.

---

## 주요 기능

### 1. ASAM (Adaptive Sharpness-Aware Minimization)
- 모델 파라미터 sharpness를 고려하여 일반화 성능 향상
- **하이퍼파라미터**
  - `--optim`: `ASAM` 선택
  - `--rho`: sharpness radius (예: 0.5)
  - `--eta`: step size (예: 0.01)
- **사용법**
```bash
--optim ASAM --rho 0.5 --eta 0.01
```

### 2. ANL (Adaptive Negative Learning)
- **개념**
	- `Positive Learning (PL)`: 정답 클래스 학습 (Cross Entropy)
	-	`Negative Learning (NL)`: 모델이 높은 확률을 준 오답 클래스 학습
	-	`Adaptive`: 혼동이 예상되는 클래스만 선택적으로 NL 수행
- **하이퍼파라미터**
	-	`--cost-type`: anl 선택
	-	`--beta`: NL 비율 조정 (예: 0.1)
- **사용법**
```bash
--cost-type anl --beta 0.1
```

### 3. Image Aware Crop & Resize (new_PetSkin)
- **개념**
	- ROI(병변) 중심으로 주변 Context 포함 224x224 크기로 Crop
	- 원본이 224x224보다 작으면 절대 확대하지 않음
	- 작은 이미지일 경우 Padding 적용
	- 큰 이미지는 Resize 처리
	- 이미지 캐싱: npy 형식으로 전처리 후 캐싱하여 학습 속도 향상
- **하이퍼파라미터**
	- 없음, ROI 기반 자동 처리
---

## 설치 및 요구사항
- PyTorch GPU 버전
- Python 3.8 이상
- 라이브러리 설치

```bash
pip install -r requirements.txt
```
- uv-astral 권장

---

## 데이터 준비
- 데이터셋 구조:

```bash
PetSkin/
├─ images/
│  ├─ class_001/
│  │  ├─ 000000.jpg
│  │  ├─ 000001.jpg
│  │  ├─ ...
│  │  ├─ 000000.json
│  │  ├─ 000001.json
│  │  ├─ ...
│  │  ├─ 000000.json
│  ├─ class_002/
│  └─ ...
```

- 이미지 전처리 후 npy 캐싱 

---

## 사용 방법

1. 스크립트 실행

```bash
chmod +x scripts/mydatasets.sh
./scripts/mydatasets.sh
```

2. 학습 커맨드 예시

```bash
python train.py \
--dataset petskin \
--dataRoot /path/to/PetSkin \
--optim ASAM --rho 0.5 --eta 0.01 \
--cost-type anl --beta 0.1 \
--forget-type coteaching_plus \
--warmup 10 \
--stage1 50 --stage2 50 \
--forget-rate 0.2 \
--num-gradual 10 \
--alpha 0.9 --lambda1 0.5 \
--cache_dir /path/to/cache
```


---

## 하이퍼파라미터 설명

| 하이퍼파라미터	| 설명	| 기본값 |
|---|---|---|
|--dataset	| 사용할 데이터셋 이름	| petskin |
|--dataRoot	| 데이터셋 루트 경로	| ./PetSkin |
|--optim	| 최적화 방법	| ASAM (권장)|
|--rho	| ASAM sharpness radius	| 0.5 (권장)|
|--eta	| ASAM step size	| 0.01 (권장)|
|--cost-type	| 손실 함수 선택 (CE, anl)	| CE (권장)|
|--beta	| ANL Negative Learning 비율	| 0.1 (권장)|
|--forget-type	| Co-Correcting 샘플 선택 전략	| coteaching_plus (권장) |
|--warmup	| Warm-up 단계 에포크 수	| 별도 조정 |
|--stage1	| Co-Correcting Stage1 에포크 수	| 별도 조정 |
|--stage2	| Co-Correcting Stage2 에포크 수	| 별도 조정 |
|--forget-rate	| 라벨 노이즈 제거 비율	| 별도 조정 |
|--num-gradual	| Forget rate 증가 단계 수	| 별도 조정 |
|--alpha	| Co-Correcting Stage1/Stage2 weighting	| ~ 0.8 (권장) |
|--lambda1	| Loss weight for Negative Learning	| ~ 400 (권장) |
|--cache_dir	npy 캐시 저장 경로	| ./cache |

---

## 데이터 캐싱
- 학습 속도 향상을 위해 npy 형식으로 이미지/라벨 캐싱
- 캐시 경로: --cache_dir로 지정
- 기존 캐시가 있으면 재생성 없이 바로 로딩

---