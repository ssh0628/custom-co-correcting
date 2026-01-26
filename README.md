# Custom Co-Correcting

이 프로젝트는 **Co-Correcting** 알고리즘을 기반으로 커스텀 데이터셋(예: PetSkin) 학습을 위한 확장된 버전. 
**ASAM Optimizer**와 **ANL (Adaptive Negative Learning)** 기능을 추가하여 노이즈가 있는 라벨 환경에서도 강력한 성능을 발휘하도록 개선

## 주요 기능 (Key Features)

### 1. ASAM (Adaptive Sharpness-Aware Minimization)
일반화 성능을 위한 **최적화 기법**
- **사용법**:
  ```bash
  --optim ASAM --rho 0.5 --eta 0.01
  ```

### 2. ANL (Adaptive Negative Learning)
ANL은 잘못된 라벨(Noisy Label)이 포함된 데이터셋에서 모델이 잘못된 정보를 학습하는 것을 방지하는 **손실 함수**
- **Positive Learning (PL)**: "이 이미지는 강아지다"라고 학습 (일반적인 Cross Entropy).
- **Negative Learning (NL)**: "이 이미지는 적어도 자동차는 아니다"라고 학습.
- **Adaptive**: 모든 클래스에 대해 NL을 수행하는 것이 아니라, 모델이 헷갈려하거나 확률이 높은 오답 클래스를 똑똑하게 선별하여 학습.
- **사용법**:
  ```bash
  --cost-type anl --beta 0.1
  ```
### 3. Image Aware Crop & Resize(new_PetSkin)
**로직**: ROI(병변)가 224x224보다 작으면? -> 절대 확대 안 함.
**동작**: 원본 이미지에서 병변 중심으로 주변 배경(Context)을 더 포함해서 224 크기로 잘라옴. (화질 보존 + 문맥 정보 확보)
**예외처리**: 원본 자체가 작으면 어쩔 수 없이 Padding 함. 큰 이미지는 그냥 Resize.


---

## 사용 방법 (Usage)
커스텀 데이터 셋일 경우 별도의 코드 수정 필요
Hyper Parameter가 많으니 Script 형식으로 실행 권장

### 1. 스크립트 실행
`scripts/mydatasets.sh`

```bash
# 실행 권한 부여
chmod +x scripts/mydatasets.sh

# 학습 시작
./scripts/mydatasets.sh
```

### 2. 주요 옵션 설명
- **`--dataset`**: 사용할 데이터셋 이름 (예: `petskin`)
- **`--dataRoot`**: 데이터셋 경로
- **`--warmup`**:
- **`--stage1`**:
- **`--stage2`**:
- **`--forget-rate`**:
- **`--num-gradual`**:
- **`--alpha`**:
- **`--lambda1`**:
- **`--beta`**:

- **`--forget-type`** (Selection Strategy):
    - `coteaching_plus`: 두 모델의 예측이 다를 때 학습 (추천)
    - `coteaching`: 손실이 작은 것만 학습
- **`--cost-type`** (Cost Function):
    - `anl`: Adaptive Negative Learning 
    - `CE`: Cross Entropy

---

## 설치 및 요구사항
- **PyTorch GPU 버전**이 필요합니다. (Windows/Linux 환경 권장)
- 필요한 라이브러리 설치:
  ```bash
  pip install -r requirements.txt
  ```
  혹은 uv-astral로 실행 (추천)