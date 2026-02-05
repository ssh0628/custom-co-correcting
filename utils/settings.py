import sys
import argparse

def get_args():
    if sys.platform == 'darwin':
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/ISIC-Archive-Downloader/Data_sample_balanced'
        mnist_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/MNIST'
        cifar10_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar10'
        cifar100_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar100'
        pcam_root = "/Users/jiarunliu/Documents/BUCT/Label_517/dataset/PatchCamelyon"
        batch_size = 8
        device = 'cpu'
        data_device = 0
        noise_type = 'sn'
        stage1 = 1
        stage2 = 3

    elif sys.platform == 'linux':
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/home/fgldlb/Documents/ISIC-Archive-Downloader/NewData'
        pcam_root = "/home/fgldlb/Documents/dataset/PatchCamelyon"
        mnist_root = './data/mnist'
        cifar10_root = './data/cifar10'
        cifar100_root = './data/cifar100'
        batch_size = 32
        device = 'cuda:0'
        data_device = 1
        noise_type = 'sn'
        stage1 = 70
        stage2 = 200

    elif sys.platform == 'win32':
        # Windows Default Settings
        isic_root = 'D:/Datasets/ISIC'
        petskin_root = 'D:/Datasets/PetSkin'
        mnist_root = './data/mnist'
        pcam_root = "C:\\Users\\cream\\OneDrive\\Desktop\\co-correcting-data"
        batch_size = 32
        device = 'cuda:0' # GPU
        data_device = 0
        noise_type = 'clean'
        stage1 = 70
        stage2 = 200

    else:
        # Default Settings
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = None
        mnist_root = './data/mnist'
        cifar10_root = '/data/cifar10'
        cifar100_root = '/data/cifar100'
        pcam_root = None
        batch_size = 16
        device = 'cpu'
        data_device = 0
        noise_type = 'clean'
        stage1 = 70
        stage2 = 200

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # General Parameters
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                        metavar='N', help='Mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='H-P', help='Initial Learning Rate')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-5, type=float,
                        metavar='H-P', help='Stage 3 (Finetuning) Learning Rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum (e.g., SGD)')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='Weight Decay (default: 1e-4)')
    parser.add_argument('--backbone', dest="backbone", default="resnet50", type=str,
                        help="Backbone network (e.g., resnet50)")
    parser.add_argument('--optim', dest="optim", default="SGD", type=str,
                        choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'Adadelta', 'Adagrad', 'mix', 'ASAM'],
                        help="Optimizer choice")
    parser.add_argument('--scheduler', dest='scheduler', default=None, type=str, choices=['cyclic', None, "SWA", "Cosine"],
                        help="LR Scheduler choice")
    # 기존 코드는 Default로 4로 설정되어 있었음
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='Workers for data loading (default: 0)')
    
    # ASAM Optimizer Parameters (Custom)
    parser.add_argument('--rho', default=0.5, type=float,
                        help='ASAM rho (Ascent Step Size).')
    parser.add_argument('--eta', default=0.01, type=float,
                        help='ASAM eta (Smoothing). Default 0.01')
    
    # Co-teaching Params
    """
        --forget rate : R(t) = 1 - forget_rate
        --num-gradual : E_k
    """
    parser.add_argument('--forget-rate', '--fr', '--forget_rate', default=0.2, type=float,
                        metavar='H-P', help='Forget rate (match noise rate).')
    parser.add_argument('--num-gradual', '--ng', '--num_gradual', default=10, type=int,
                        metavar='H-P', help='Epochs to linearly increase forget rate.')
    parser.add_argument('--exponent', default=1, type=float,
                        metavar='H-P', help='Forget rate increase exponent (1=linear).')
    parser.add_argument('--forget-type', dest="forget_type", default="coteaching_plus", type=str,
                        choices=['coteaching_plus', 'coteaching'],
                        help="Selection Strategy: [coteaching_plus, coteaching]")
    parser.add_argument('--cost-type', dest="cost_type", default="CE", type=str,
                        choices=['CE', 'anl'],
                        help="Cost Function: [CE, anl]")
    parser.add_argument('--warmup', '--wm', '--warm-up', default=0, type=int,
                        metavar='H-P', help='Warmup epochs (trust all data).')
    parser.add_argument('--linear-num', '--linear_num', default=256, type=int,
                        metavar='H-P', help='Linear layer nodes (specific models).')
    
    # PENCIL Parameters
    """
        --alpha : compatibility loss weight
        --beta : entropy loss weight
        --lambda1 : label correction rate
    """
    parser.add_argument('--alpha', default=0.4, type=float,
                        metavar='H-P', help='Compatibility Loss coeff (alpha)')
    parser.add_argument('--beta', default=0.1, type=float,
                        metavar='H-P', help='Entropy Loss coeff (beta)')
    parser.add_argument('--lambda1', default=200, type=int,
                        metavar='H-P', help='Label Correction strength (lambda). Higher = less trust in original.')
    parser.add_argument('--K', default=10.0, type=float, )
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='Start epoch number')
    parser.add_argument('--epochs', default=320, type=int, metavar='H-P',
                        help='Total epochs')
    parser.add_argument('--stage1', default=stage1, type=int,
                        metavar='H-P', help='Stage 1 end epoch (Warmup/Initial)')
    parser.add_argument('--stage2', default=stage2, type=int,
                        metavar='H-P', help='Stage 2 end epoch (Label Correction)')
    
    # 노이즈(Noise) 설정
    parser.add_argument('--noise', default=0.20, type=float,
                        help='Label noise ratio (experimental)')
    parser.add_argument('--noise_type', default=noise_type,  choices=['clean', 'sn', 'pairflip'],type=str,
                        help='Noise type: clean, sn (symmetric), pairflip')
    
    # 데이터(Data) 설정
    parser.add_argument("--dataset", dest="dataset", default='mnist', type=str,
                        choices=['mnist', 'cifar10', 'cifar100', 'cifar2', 'isic', 'clothing1m', 'pcam', 'petskin'],
                        help="Dataset choice (includes petskin)")
    parser.add_argument("--image_size", dest="image_size", default=224, type=int,
                        help="Input image size (e.g., 224)")
    parser.add_argument('--classnum', default=8, type=int,
                        metavar='H-P', help='Number of classes')
    parser.add_argument('--device', dest='device', default=device, type=str,
                        help='Device (e.g., cuda:0)')
    parser.add_argument('--data_device', dest="data_device", default=data_device, type=int,
                        help="Data location: 0=disk, 1=RAM (0 recommended if OOM).")
    parser.add_argument('--dataRoot',dest='root',default=isic_root,
                        type=str,metavar='PATH',help='Dataset path')
    parser.add_argument('--datanum', default=22528, type=int,
                        metavar='H-P', help='Training sample count')
    parser.add_argument('--train-redux', dest="train_redux", default=None, type=int,
                        help='train data number, default None')
    parser.add_argument('--test-redux', dest="test_redux", default=None, type=int,
                        help='test data number, default None')
    parser.add_argument('--val-redux', dest="val_redux", default=None, type=int,
                        help='validate data number, default None')
    parser.add_argument('--full-test', dest="full_test", default=False, type=bool,
                        help='use full test set data, default False')
    parser.add_argument('--random-ind-redux', dest="random_ind_redux", default=False, type=bool,
                        help='use full test set data, default False')
    # Curriculum settings
    parser.add_argument("--curriculum", dest="curriculum", default=1, type=int,
                        help="curriculum in label updating")
    parser.add_argument("--cluster-mode", dest="cluster_mode", default='dual', type=str, choices=['dual', 'single', 'dual_PCA'],
                        help="curriculum in label updating")
    parser.add_argument("--dim-reduce", dest="dim_reduce", default=256, type=int,
                        help="Curriculum features dim reduce by PCA")
    parser.add_argument("--mix-grad", dest="mix_grad", default=1, type=int,
                        help="mix gradient of two-stream arch, 1=True")
    parser.add_argument("--discard", dest="discard", default=0, type=int,
                        help="only update discard sample's label, 1=True")
    parser.add_argument("--gamma", dest="gamma", default=0.6, type=int,
                        help="forget rate schelduler param")
    parser.add_argument("--finetune-schedule", '-fs', dest="finetune_schedule", default=0, type=int,
                        help="forget rate schelduler param")
    # trainer settings
    parser.add_argument('--dir', dest='dir', default="experiment/test-debug", type=str,
                        metavar='PATH', help='save dir')
    parser.add_argument('--random-seed', dest='random_seed', default=None, type=int,
                        metavar='N', help='pytorch random seed, default None.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    # ConvNext settings
    parser.add_argument('--drop-path-rate', dest='drop_path_rate', default=0.1, type=float,
                        metavar='N', help='drop path rate, default 0.1')
    parser.add_argument("--convnext", dest="convnext", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str,
                        help="use convnext, default convnextv2_tiny.fcmae_ft_in22k_in1k")

    # Diversity Regularization Settings
    parser.add_argument('--diversity-lambda-warmup', dest='diversity_lambda_warmup', default=0.1, type=float,
                        metavar='L', help='Diversity lambda for Warm-up stage (default: 0.1)')
    parser.add_argument('--diversity-lambda-stage1', dest='diversity_lambda_stage1', default=0.5, type=float,
                        metavar='L', help='Diversity lambda for Stage 1 (default: 0.5)')
    parser.add_argument('--diversity-lambda-stage2', dest='diversity_lambda_stage2', default=0.1, type=float,
                        metavar='L', help='Diversity lambda for Stage 2 (label correction) (default: 0.1)')
    parser.add_argument('--diversity-lambda-finetune', dest='diversity_lambda_finetune', default=0.0, type=float,
                        metavar='L', help='Diversity lambda for Fine-tuning stage (default: 0.0)')

    # Pretrained Flag
    parser.add_argument("--pretrained", dest="pretrained", default=1, type=int,
                        help="1: use pretrained backbone recipe, 0: scratch")

    parser.add_argument("--freeze-warmup", dest="freeze_warmup", default=-1, type=int,
                        help="ConvNeXt warmup behavior. -1: auto, 0: no freeze, 1: freeze backbone")
    
    args = parser.parse_args()

    # Setting for different dataset
    if args.dataset == "isic":
        print("Training on ISIC")
        args.backbone = 'resnet50'
        args.image_size = 224
        args.classnum = 2
        args.input_dim = 3

    elif args.dataset == 'mnist':
        print("Training on mnist")
        args.backbone = 'cnn'
        if args.root == isic_root:
            args.root = mnist_root
        args.batch_size = 128
        args.image_size = 28
        args.classnum = 10
        args.input_dim = 1
        args.linear_num = 144
        args.datanum = 60000
        args.lr = 0.001
        args.lr2 = 0.0001

    elif args.dataset == 'pcam':
        if args.root == isic_root:
            args.root = pcam_root
        args.backbone = 'densenet169'
        args.batch_size = 128
        args.image_size = 96
        args.dim_reduce = 128
        args.classnum = 2
        args.input_dim = 3
        args.stage1 = 1
        args.stage2 = 2
        args.epochs = 3 # 테스트 용으로 설정함
        args.num_gradual = 3 # 테스트 용으로 설정함
        args.datanum = 262144
        # args.train_redux = 26214
        # args.test_redux = 3276
        # args.val_redux = 3276
        # args.val_redux = 3276
        args.random_ind_redux = False

    elif args.dataset == 'petskin':
        print("Training on PetSkin")
        # args.backbone = 'resnet50' # 기본 모델: ResNet50 (Deleted to allow user input)
        args.image_size = 224 # 이미지 크기
        args.classnum = 8 # 클래스 개수 (A1~A6)
        args.input_dim = 3 # 입력 채널 (RGB)
        # 기본 데이터 개수 설정 (실제 데이터셋 크기에 따라 다를 수 있음)
        args.datanum = 22528



    else:
        print("Use default setting")

    return args