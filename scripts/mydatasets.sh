# ==============================================================================
# MyDatasets Execution Script (Custom Dataset Template)
# ==============================================================================
# 이 스크립트는 사용자 커스텀 데이터셋(예: PetSkin)을 학습시키기 위한 템플릿입
# 모든 하이퍼파라미터에 대한 설명과 기본값, 추천 설정이 포함되어 있습니다.
#
# 사용 방법:
# 1. 터미널(Git Bash, WSL, Linux 등)에서 실행 권한 부여: chmod +x scripts/mydatasets.sh
# 2. 실행: ./scripts/mydatasets.sh
# ==============================================================================

# ------------------------------------------------------------------------------
# 기본 설명 (Basic Description)
# ------------------------------------------------------------------------------
# python Co-Correcting.py : 메인 학습 스크립트 실행

python Co-Correcting.py \
    \
    # --------------------------------------------------------------------------
    # 1. 데이터셋 및 경로 설정 (Dataset & Path Settings)
    # --------------------------------------------------------------------------
    # [필수 수정] 사용할 데이터셋 이름입니다.
    # 옵션: 'mnist', 'cifar10', 'cifar100', 'cifar2', 'isic', 'clothing1m', 'pcam', 'petskin'
    # Default: 'mnist'
    --dataset petskin \
    \
    # [필수 수정] 데이터셋이 위치한 루트 디렉토리 경로입니다.
    # Default: OS별로 다름 (settings.py 참조), 없으면 None
    # 윈도우 경로 예시: "D:/Datasets/PetSkin"
    --dataRoot "D:/Datasets/PetSkin" \
    \
    # [필수 수정] 실험 결과를 저장할 디렉토리 경로입니다.
    # Default: "experiment/test-debug"
    --dir "experiment/myset/petskin_run1" \
    \
    # [중요] 전체 학습 데이터 샘플 수입니다. 정확한 수치를 입력해야 합니다.
    # PetSkin의 경우 약 32000 등으로 예상되나 실제 데이터 수를 확인 후 입력하세요.
    # Default: 15000 (MNIST 등 기본값)
    --datanum 32000 \
    \
    # 데이터셋의 클래스(카테고리) 개수입니다.
    # PetSkin: 6 (A1~A6)
    # Default: 2
    --classnum 6 \
    \
    # 입력 이미지의 크기입니다. (가로x세로가 같다고 가정)
    # ResNet50 백본 사용 시 보통 224를 사용합니다.
    # Default: 224
    --image_size 224 \
    \
    # --------------------------------------------------------------------------
    # 2. 모델 및 하드웨어 설정 (Model & Hardware Settings)
    # --------------------------------------------------------------------------
    # 사용할 백본 네트워크 모델입니다.
    # Default: "resnet50"
    --backbone resnet50 \
    \
    # 사용할 GPU/CPU 디바이스입니다.
    # Default: 'cuda:0' (Linux/Win), 'cpu' (Mac)
    --device cuda:0 \
    \
    # 데이터를 어디서 로드할지 설정합니다.
    # 0: 디스크에서 실시간 읽기 (메모리 절약, 속도 약간 느림) - 추천
    # 1: RAM에 미리 로드 (메모리 많이 씀, 속도 빠름)
    # Default: 0
    --data_device 0 \
    \
    # 데이터 로딩에 사용할 워커 프로세스 수입니다.
    # 윈도우에서는 오류 방지를 위해 0을 권장합니다. 리눅스라면 4 이상 가능.
    # Default: 0
    --workers 0 \
    \
    # --------------------------------------------------------------------------
    # 3. 학습 기본 파라미터 (Basic Training Parameters)
    # --------------------------------------------------------------------------
    # 총 학습 에폭(Epoch) 수입니다.
    # Co-Correcting은 충분한 학습이 필요하므로 보통 200~300 이상을 권장합니다.
    # Default: 320
    --epochs 320 \
    \
    # 미니 배치 크기입니다. GPU 메모리에 맞춰 조절하세요.
    # Default: 16 (Linux 32)
    --batch-size 32 \
    \
    # 초기 학습률 (Initial Learning Rate)입니다.
    # Default: 1e-4
    --lr 1e-4 \
    \
    # Stage 3 (파인튜닝) 단계에서의 학습률입니다. 더 세밀한 조정이 필요하므로 낮게 설정합니다.
    # Default: 1e-5
    --lr2 1e-5 \
    \
    # 최적화 알고리즘 (Optimizer) 선택
    # 옵션: 'SGD', 'Adam', 'ASAM' (Custom)
    # [추천] 커스텀 ASAM을 사용하려면 'ASAM'으로 설정하세요.
    --optim ASAM \
    \
    # [ASAM 전용] rho 파라미터 (Ascent Step 크기)
    # Default: 0.5
    --rho 0.5 \
    \
    # [ASAM 전용] eta 파라미터
    # Default: 0.01
    --eta 0.01 \
    \
    # 모멘텀 (Momentum), SGD/ASAM 사용 시 주로 사용됩니다.
    # Default: 0.9
    --momentum 0.9 \
    \
    # 가중치 감쇠 (Weight Decay)
    # Default: 1e-3
    --weight-decay 1e-3 \
    \
    # 학습률 스케줄러 선택
    # 옵션: 'cyclic', None, 'SWA'
    # Default: None
    --scheduler None \
    \
    # 시드(Seed) 고정 (재현성 확보용)
    # Default: None
    # --random-seed 42 \
    \
    # --------------------------------------------------------------------------
    # 4. 노이즈 및 Co-Correction 핵심 파라미터 (Noise & Co-Correction Core)
    # --------------------------------------------------------------------------
    # 데이터 라벨의 노이즈 비율 (실험용 인위적 노이즈 주입 시 사용).
    # 리얼 월드 데이터셋(PetSkin 등)을 그대로 쓸 때는 0.0 (clean)이나 낮은 값으로 설정.
    # Default: 0.20
    --noise 0.0 \
    \
    # 노이즈 타입
    # 옵션: 'clean' (노이즈 없음), 'sn' (대칭 노이즈), 'pairflip' (페어 플립)
    # 커스텀 데이터셋이 이미 노이즈가 섞인 리얼 데이터라면 'clean'으로 두고
    # 알고리즘이 스스로 필터링하게 하거나, 실험 세팅에 맞춰 설정하세요.
    # Default: 'sn'
    --noise_type clean \
    \
    # 샘플 선택 방식 (Selection Strategy)
    # 옵션: 'coteaching_plus' (Disagreement), 'coteaching' (Standard)
    # Default: "coteaching_plus"
    --forget-type coteaching_plus \
    \
    # 비용 함수 (Cost Function)
    # 옵션: 'anl' (Adaptive Negative Learning), 'CE' (Cross Entropy)
    # Default: "CE"
    # [추천] 커스텀 Adaptive Negative Learning을 사용하려면 'anl'로 설정하세요.
    --cost-type anl \
    \
    # 망각(Forget) 비율. 학습 데이터 중 '노이즈'라고 판단하여 버릴 비율의 추정치입니다.
    # 실제 노이즈 비율과 비슷하게 설정하는 것을 권장합니다.
    # Default: 0.2
    --forget-rate 0.2 \
    \
    # 망각 비율을 0에서 목표치(forget-rate)까지 선형적으로 높여갈 에폭 수입니다.
    # Default: 10
    --num-gradual 10 \
    \
    # 망각 비율 증가 지수 (1이면 선형, 1보다 크면 초반에 천천히 증가)
    # Default: 1
    --exponent 1 \
    \
    # --------------------------------------------------------------------------
    # 5. 스테이지 설정 (Stage Settings - Co-Correcting Phases)
    # --------------------------------------------------------------------------
    # Stage 1: Warm-up 및 초기 학습이 종료되는 에폭입니다.
    # 이 시점 전까지는 일반 학습(또는 Warmup)을 수행합니다.
    # Default: 70
    --stage1 70 \
    \
    # Stage 2: 라벨 수정(Label Correction) 단계가 종료되는 에폭입니다.
    # 이 시점 이후에는 확정된 라벨로 파인튜닝(Stage 3)을 진행합니다.
    # Default: 200
    --stage2 200 \
    \
    # 웜업(Warm up) 에폭 수. 초기에는 모든 데이터를 신뢰하고 학습합니다.
    # Default: 0
    --warmup 10 \
    \
    # --------------------------------------------------------------------------
    # 6. PENCIL / Label Correction 파라미터 (Advanced)
    # --------------------------------------------------------------------------
    # 라벨 수정(Label Correction) 강도 (lambda).
    # 값이 클수록 원래 라벨(target)의 영향력이 커집니다. (즉, 수정이 보수적으로 됨?? -> 코드 확인 필요)
    # *코드 로직 상: lambda1 * grad 만큼 라벨 분포(yy)를 업데이트함.*
    # Default: 200
    --lambda1 200 \
    \
    # 호환성 손실(Compatibility Loss)의 가중치 (Alpha).
    # 모델의 예측(output)과 라벨 분포(yy)가 얼마나 일치해야 하는지 조절합니다.
    # Default: 0.4
    --alpha 0.4 \
    \
    # 엔트로피 손실(Entropy Loss)의 가중치 (Beta).
    # 라벨 분포가 One-hot(확실한 라벨)에 가까워지도록 유도합니다.
    # Default: 0.1
    --beta 0.1 \
    \
    # --------------------------------------------------------------------------
    # 7. 커리큘럼 및 클러스터링 설정 (Curriculum & Clustering - Advanced)
    # --------------------------------------------------------------------------
    # 라벨 업데이트 시 커리큘럼 학습 사용 여부 (1=True, 0=False)
    # Default: 1
    --curriculum 1 \
    \
    # 클러스터링 모드 (Curriculum용 특징 추출 방식)
    # 옵션: 'dual' (두 모델 특징 결합), 'single', 'dual_PCA'
    # Default: 'dual'
    --cluster-mode dual \
    \
    # PCA 사용 시 축소할 차원 수
    # Default: 256
    --dim-reduce 256 \
    \
    # Two-Stream 구조에서 그래디언트 혼합 여부 (1=True)
    # Default: 1
    --mix-grad 1 \
    \
    # Discard된 샘플의 라벨만 업데이트할지 여부 (1=True)
    # Default: 0 (모든 샘플 라벨 업데이트 시도)
    --discard 0 \
    \
    # Forget Rate 스케줄러 파라미터 (Gamma) - 파인튜닝 시 사용
    # Default: 0.6
    --gamma 0.6 \
    \
    # 파인튜닝 스케줄 사용 여부 (Switch)
    # Default: 0
    --finetune-schedule 0 \
    \
    # --------------------------------------------------------------------------
    # 8. 기타/디버깅 설정 (Misc & Debugging)
    # --------------------------------------------------------------------------
    # 선형 레이어의 노드 수 (MNIST 등 일부 모델 구조용, ResNet에서는 무시될 수 있음)
    # Default: 256
    --linear-num 256 \
    \
    # K 값 (One-hot 벡터 생성 시 스케일링/초기화 관련 파라미터로 추정)
    # Default: 10.0
    --K 10.0 \
    \
    # 시작 에폭 번호 (체크포인트에서 이어서 학습할 때 사용)
    # Default: 0
    --start-epoch 0 \
    \
    # [데이터 수동 축소 - 테스트용]
    # --train-redux 1000 \
    # --test-redux 100 \
    # --val-redux 100 \
    \
    # 전체 테스트 셋 사용 여부
    # Default: False
    --full-test False \
    \
    # 랜덤 인덱스 불일치 사용 여부 (테스트용)
    # Default: False
    --random-ind-redux False

# ==============================================================================
# 실행 끝
# ==============================================================================
