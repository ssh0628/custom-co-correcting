# Custom Co-Correcting

이 프로젝트는 **Co-Correcting (Co-Pencil)** 알고리즘을 기반으로 커스텀 데이터셋(예: PetSkin) 학습을 위해 확장된 버전입니다. 특히 **ASAM Optimizer**와 **ANL (Adaptive Negative Learning)** 기능을 추가하여 노이즈가 있는 라벨 환경에서도 강력한 성능을 발휘하도록 개선되었습니다.

## 주요 기능 (Key Features)

### 1. ASAM (Adaptive Sharpness-Aware Minimization)
ASAM은 모델의 일반화 성능을 극대화하기 위한 **고급 최적화 기법**입니다.
- **개념**: 손실 함수(Loss Landscape)의 **평탄한 최소점(Flat Minima)**을 찾습니다. 평탄한 곳에 위치한 모델은 테스트 데이터가 약간 변하더라도 성능이 급격히 떨어지지 않아 일반화 성능이 뛰어납니다.
- **Adaptive**: 기존 SAM과 달리 파라미터의 스케일(크기)에 따라 섭동(Perturbation)의 크기를 조절하여 더 안정적이고 강력한 학습이 가능합니다.
- **사용법**:
  ```bash
  --optim ASAM --rho 0.5 --eta 0.01
  ```

### 2. ANL (Adaptive Negative Learning)
ANL은 잘못된 라벨(Noisy Label)이 포함된 데이터셋에서 모델이 잘못된 정보를 학습하는 것을 방지하는 **손실 함수**입니다.
- **Positive Learning (PL)**: "이 이미지는 강아지다"라고 학습 (일반적인 Cross Entropy).
- **Negative Learning (NL)**: "이 이미지는 적어도 자동차는 아니다"라고 학습.
- **Adaptive**: 모든 클래스에 대해 NL을 수행하는 것이 아니라, 모델이 헷갈려하거나 확률이 높은 오답 클래스를 똑똑하게 선별하여 학습합니다. 이는 노이즈 라벨의 악영향을 획기적으로 줄여줍니다.
- **Co-teaching+ 결합**: 본 프로젝트에서는 ANL을 Co-teaching+의 **불일치(Disagreement)** 전략과 결합하여, 두 모델이 서로 동의하지 않는 어려운 샘플에 대해 더욱 정교하게 학습합니다.
- **사용법**:
  ```bash
  --cost-type anl --beta 0.1
  ```

---

## 사용 방법 (Usage)

커스텀 데이터셋 학습을 위한 템플릿 스크립트가 준비되어 있습니다.

### 1. 스크립트 실행
`scripts/mydatasets.sh` 파일에는 모든 하이퍼파라미터에 대한 상세한 설명이 한글 주석으로 달려 있습니다.

```bash
# 실행 권한 부여
chmod +x scripts/mydatasets.sh

# 학습 시작
./scripts/mydatasets.sh
```

### 2. 주요 옵션 설명
- **`--dataset`**: 사용할 데이터셋 이름 (예: `petskin`)
- **`--dataRoot`**: 데이터셋 경로
- **`--forget-type`** (Selection Strategy):
    - `coteaching_plus`: 두 모델의 예측이 다를 때 학습 (추천)
    - `coteaching`: 손실이 작은 것만 학습
- **`--cost-type`** (Cost Function):
    - `anl`: Adaptive Negative Learning (추천)
    - `CE`: Cross Entropy

---

## 설치 및 요구사항
- **PyTorch GPU 버전**이 필요합니다. (Windows/Linux 환경 권장)
- 필요한 라이브러리 설치:
  ```bash
  pip install -r requirements.txt
  ```
  혹은 uv-astral로 실행 (추천)