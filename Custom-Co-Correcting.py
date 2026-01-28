import os
import copy
import json
import time
import shutil
import numpy as np

from os.path import join

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from utils.settings import get_args
from utils.label_checker import check_label
from utils.curriculum_clustering import CurriculumClustering
from utils.asam import ASAM

from Loss import Loss
from BasicTrainer import BasicTrainer


class CoCorrecting(BasicTrainer, Loss):
    """
    Train Co-Pencil Method
    
    Co-Correcting (또는 Co-Pencil) 학습 클래스
    - BasicTrainer: 학습 환경 설정, 로깅, 모델 저장 등의 기본 기능 상속
    - Loss: Co-teaching 및 PENCIL 손실 함수 계산 로직 상속
    """

    def __init__(self, args):
        super().__init__(args)
        """
            model A, B : Dual architecture (상호 학습을 위한 두 개의 모델)
            model C : best model (최고 성능 모델 저장용)
            prepare optim, loss func (최적화 도구 및 손실 함수 준비)
        """
        # 상호 학습을 위한 모델 초기화 (동일한 구조의 모델 A, B 생성)

        # Model A Initialization (Seed: Base)
        if self.args.random_seed is not None:
            torch.manual_seed(self.args.random_seed)
            # torch.cuda.manual_seed_all(self.args.random_seed)
        self.modelA = self._get_model(self.args.backbone)
        
        # Model B Initialization (Seed: Base + 1)
        if self.args.random_seed is not None:
            torch.manual_seed(self.args.random_seed)
            # torch.manual_seed(self.args.random_seed + 1)
            # torch.cuda.manual_seed_all(self.args.random_seed + 1)
        self.modelB = self._get_model(self.args.backbone)

        # 데이터 로더의 순서는 동일해야 하므로 Base Seed로 복구
        if self.args.random_seed is not None:
            torch.manual_seed(self.args.random_seed)
            # torch.cuda.manual_seed_all(self.args.random_seed)

        self.modelC = self._get_model(self.args.backbone)

        # freeze all layers except head
        for name, param in self.modelA.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        for name, param in self.modelB.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        
        # Optimizer & Criterion (최적화 함수 및 기준)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.logsoftmax = nn.LogSoftmax(dim=1).to(self.args.device)
        self.softmax = nn.Softmax(dim=1).to(self.args.device)

        # ASAM은 기본 옵티마이저(SGD 등)를 감싸서 동작하므로, 
        # 초기화 시에는 SGD로 생성한 후 ASAM으로 래핑
        # ASAM Optimizer Wrapping
        if self.args.optim == 'ASAM':
            print("Initializing ASAM Optimizer...")
            # ASAM은 Base Optimizer(SGD 등)를 감싸서 동작함
            rho = getattr(self.args, 'rho', 0.5)
            eta = getattr(self.args, 'eta', 0.01)
            
            # 1. Base Optimizer (SGD) 생성
            self.optimizerA = self._get_optim(self.modelA.parameters(), optim='SGD')
            self.optimizerB = self._get_optim(self.modelB.parameters(), optim='SGD')
            
            # 2. Wrap with ASAM
            self.optimizerA = ASAM(self.optimizerA, self.modelA, rho=rho, eta=eta)
            self.optimizerB = ASAM(self.optimizerB, self.modelB, rho=rho, eta=eta)
        else:
            # Normal Optimizer check
            self.optimizerA = self._get_optim(self.modelA.parameters(), optim=self.args.optim)
            self.optimizerB = self._get_optim(self.modelB.parameters(), optim=self.args.optim)

        # Cosine Scheduler
        if self.args.scheduler == "Cosine":
            # Phase 1: lr -> lr2 (until Stage 1)
            self.schedulerA = CosineAnnealingLR(self.optimizerA, T_max=self.args.stage1, eta_min=self.args.lr2)
            self.schedulerB = CosineAnnealingLR(self.optimizerB, T_max=self.args.stage1, eta_min=self.args.lr2)
        
        # 체크 포인트 불러오기
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print(f"=> loading checkpoint '{self.args.resume}'")
                checkpoint = torch.load(self.args.resume)
                self.start_epoch = checkpoint['epoch']
                
                self.modelA.load_state_dict(checkpoint['state_dict_A'])
                self.optimizerA.load_state_dict(checkpoint['optimizer_A'])
                self.modelB.load_state_dict(checkpoint['state_dict_B'])
                self.optimizerB.load_state_dict(checkpoint['optimizer_B'])
                
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.resume}'")

        # trainer init
        self._recoder_init()
        self._save_meta()
        # Load Data
        self.trainloader, self.testloader, self.valloader = self._load_data()
        # Optionally resume from a checkpoint
        if os.path.isfile(self.args.checkpoint_dir):
            self._resume()
        else:
            print("=> no checkpoint found at '{}'".format(self.args.checkpoint_dir))
            # save clean label & noisy label
            np.save(join(self.args.dir, 'y_clean.npy'), self.clean_labels)

    def _resume(self):
        # load model state
        print("=> loading checkpoint '{}'".format(self.args.checkpoint_dir))
        checkpoint = torch.load(self.args.checkpoint_dir)
        self.args.start_epoch = checkpoint['epoch']
        self.best_prec1 = checkpoint['best_prec1']
        self.modelA.load_state_dict(checkpoint['state_dict_A'])
        self.optimizerA.load_state_dict(checkpoint['optimizer_A'])
        self.modelB.load_state_dict(checkpoint['state_dict_B'])
        self.optimizerB.load_state_dict(checkpoint['optimizer_B'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(self.args.checkpoint_dir, checkpoint['epoch']))

        epoch = self.args.start_epoch
        if epoch < self.args.warmup:
            # Stage 1: backbone freeze
            print("=> Resume: Stage 1 (Backbone Frozen)")
            for model in [self.modelA, self.modelB]:
                for name, p in model.named_parameters():
                    if "head" not in name:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
        else:
            # Stage 2 이후: backbone unfreeze
            print("=> Resume: Stage 2+ (Backbone Unfrozen)")
            for model in [self.modelA, self.modelB]:
                for p in model.parameters():
                    p.requires_grad = True

        # load record_dict
        if os.path.isfile(self.args.record_dir):
            print("=> loading record file {}".format(self.args.record_dir))
            with open(self.args.record_dir, 'r') as f:
                self.record_dict = json.load(f)
                print("=> loaded record file {}".format(self.args.record_dir))

    def _recoder_init(self):
        os.makedirs(self.args.dir, exist_ok=True)
        os.makedirs(join(self.args.dir, 'record'), exist_ok=True)

        keys = ['acc', 'acc5', 'label_accu', 'loss', "pure_ratio", "label_n2t", "label_t2n", "pure_ratio_discard", "margin_accu"]
        record_infos = {}
        for k in keys:
            record_infos[k] = []
        # 3 is mixed model result
        self.record_dict = {
            'train1': copy.deepcopy(record_infos),
            'test1': copy.deepcopy(record_infos),
            'val1': copy.deepcopy(record_infos),
            'train2': copy.deepcopy(record_infos),
            'test2': copy.deepcopy(record_infos),
            'val2': copy.deepcopy(record_infos),
            'train3': copy.deepcopy(record_infos),
            'test3': copy.deepcopy(record_infos),
            'val3': copy.deepcopy(record_infos),
            'loss_val': [],
            'loss_avg': [],
        }

    def _record(self):
        # write file
        with open(self.args.record_dir, 'w') as f:
            json.dump(self.record_dict, f, indent=4, sort_keys=True)

    # define drop rate schedule
    def _update_drop_path(self, model, drop_path_rate):
        for module in model.modules():
            if 'DropPath' in module.__class__.__name__:
                module.drop_prob = drop_path_rate

    def gen_forget_rate(self, forget_rate, fr_type='type_1'):
        if fr_type == 'type_1':
            rate_schedule = np.ones(args.n_epoch) * forget_rate
            rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

        if fr_type == 'type_2':
            rate_schedule = np.ones(args.n_epoch) * forget_rate
            rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
            rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2 * forget_rate,
                                                           args.n_epoch - args.num_gradual)

        return rate_schedule

    def _rate_schedule(self, epoch):
        rate_schedule = np.ones(self.args.epochs) * self.args.forget_rate
        if self.args.warmup > 0:
            rate_schedule[:self.args.warmup] = 0
            rate_schedule[self.args.warmup:self.args.warmup+self.args.num_gradual] = np.linspace(self.args.warmup,
                                                                self.args.warmup + (
                                                                            self.args.forget_rate ** self.args.exponent),
                                                                self.args.num_gradual)
        else:
            rate_schedule[:self.args.num_gradual] = np.linspace(0,
                                                                self.args.forget_rate ** self.args.exponent,
                                                                self.args.num_gradual)
        if self.args.finetune_schedule == 1:
            rate_schedule[self.args.stage2:] = np.linspace(self.args.forget_rate ** self.args.exponent,
                                                           self.args.forget_rate ** self.args.gamma,
                                                           self.args.epochs - self.args.stage2)
        return rate_schedule[epoch]

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate"""
        if epoch < self.args.stage2:
            lr = self.args.lr
        elif epoch < (self.args.epochs - self.args.stage2) // 3 + self.args.stage2:
            lr = self.args.lr2
        elif epoch < 2 * (self.args.epochs - self.args.stage2) // 3 + self.args.stage2:
            lr = self.args.lr2 / 10
        else:
            lr = self.args.lr2 / 100
        for param_group in self.optimizerA.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizerB.param_groups:
            param_group['lr'] = lr

    def _compute_loss(self, outputA, outputB, target, target_var, index, epoch, i, parallel=False):
        # 손실 계산 함수 (핵심 로직)
        
        # Warm Up 단계
        if epoch < self.args.stage1:
            # y_tilde 초기화: 초반에는 주어진 라벨(target)을 그대로 사용
            onehot = torch.zeros(
                len(target),
                self.args.classnum,
                device=target_var.device
            )

            onehot.scatter_(1, target_var.long().view(-1, 1), self.args.K)

            onehot = onehot.cpu().numpy()
            self.new_y[index, :] = onehot
            
            # Calculate Standard CE Loss for ALL samples
            # No Co-teaching selection here. Just simple training constraints.
            lossA = self._get_loss(outputA, target_var, loss_type="CE")
            lossB = self._get_loss(outputB, target_var, loss_type="CE")
            
            # All indices are "updated", none are "discarded"
            ind_A_discard = np.array([], dtype=int)
            ind_B_discard = np.array([], dtype=int)
            
            # All samples are used for update
            num_samples = target.size(0)
            ind_A_update = np.arange(num_samples)
            ind_B_update = np.arange(num_samples)
            
            # Statistics for logging (Pure Ratio is based on noise_or_not for monitoring)
            # Since we pick ALL, the pure ratio is just the dataset's pure ratio in this batch
            pure_ratio_1 = np.sum(self.noise_or_not[index]) / float(num_samples)
            pure_ratio_2 = np.sum(self.noise_or_not[index]) / float(num_samples)
            pure_ratio_discard_1 = 0.0
            pure_ratio_discard_2 = 0.0

            return lossA, lossB, onehot, onehot, ind_A_discard, ind_B_discard, ind_A_update, ind_B_update, \
                   pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2
        
        # Label Correction 단계 (Stage 1 ~ Stage 2)
        # - PENCIL 손실 함수 사용: 라벨 분포(y_tilde)를 학습 통해 업데이트
        elif epoch < self.args.stage2:
            # selection 된 데이터만 파라미터 업데이트, 나머지는 라벨만 업데이트
            yy_A = self.yy
            yy_B = self.yy
            yy_A = torch.tensor(yy_A[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            yy_B = torch.tensor(yy_B[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            # 라벨 분포(y_hat) 획득
            last_y_var_A = self.softmax(yy_A)
            last_y_var_B = self.softmax(yy_B)
            # 샘플 정렬 및 선택 (Small Loss Trick)
            forget_rate = self._rate_schedule(epoch)
            if self.args.forget_type == 'coteaching':
                # [선택 전략] Co-teaching
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, \
                pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching(
                    outputA, outputB, last_y_var_A, last_y_var_B, forget_rate, ind=index, loss_type="PENCIL",
                    target_var=target_var, noise_or_not=self.noise_or_not, parallel=parallel, softmax=False)
            elif self.args.forget_type == 'coteaching_plus':
                # [선택 전략] Co-teaching+ (Disagreement Strategy)
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, \
                pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching_plus(
                    outputA, outputB, last_y_var_A, last_y_var_B, forget_rate, epoch * i, index, loss_type="PENCIL",
                    target_var=target_var, noise_or_not=self.noise_or_not, parallel=parallel, softmax=False)
            else:
                raise NotImplementedError("forget_type {} not been found".format(self.args.forget_type))
            return lossA, lossB, yy_A, yy_B, ind_A_discard, ind_B_discard, ind_A_update, ind_B_update, \
                   pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2
        
        # Fine Tuning 단계 (Stage 2 이후)
        # - 수정된 라벨을 바탕으로 최종 미세 조정
        else:
            yy_A = self.yy
            yy_A = torch.tensor(yy_A[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            yy_B = self.yy
            yy_B = torch.tensor(yy_B[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            last_y_var_A = self.softmax(yy_A)
            last_y_var_B = self.softmax(yy_B)
            forget_rate = self._rate_schedule(epoch)
            if self.args.forget_type == 'coteaching':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching_plus(outputA, outputB,
                                                                                       last_y_var_A, last_y_var_B,
                                                                                       forget_rate,
                                                                                       ind=index,
                                                                                       loss_type="PENCIL_KL",
                                                                                       noise_or_not=self.noise_or_not,
                                                                                       softmax=False)
            elif self.args.forget_type == 'coteaching_plus':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching_plus(outputA, outputB,
                                                                                       last_y_var_A, last_y_var_B,
                                                                                       forget_rate, epoch * i,
                                                                                       index,
                                                                                       loss_type="PENCIL_KL",
                                                                                       noise_or_not=self.noise_or_not,
                                                                                       softmax=False)
            else:
                raise NotImplementedError("forget_type {} not been found".format(self.args.forget_type))
 
            return lossA, lossB, yy_A, yy_B, ind_A_discard, ind_B_discard, ind_A_update, ind_B_update, \
                   pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2

    def _gen_clustering_data(self, mode='dual'):
        if mode == 'dual':
            featureA = []
            def hookA(module, input, output):
                featureA.append(output.clone().cpu().detach())

            featureB = []
            def hookB(module, input, output):
                featureB.append(output.clone().cpu().detach())

            self.modelA.eval()
            with torch.no_grad():
                if 'convnext' in self.args.backbone.lower():
                    handleA = self.modelA.head.flatten.register_forward_hook(hookA)
                    handleB = self.modelB.head.flatten.register_forward_hook(hookB)
                else:
                    layer_num = 0
                    for i in self.modelA.modules():
                        layer_num += 1
                    target_layer_ind = layer_num - 2
                    for i, j in enumerate(self.modelA.modules()):
                        if i == target_layer_ind:
                            handleA = j.register_forward_hook(hookA)
                    for i, j in enumerate(self.modelB.modules()):
                        if i == target_layer_ind:
                            handleB = j.register_forward_hook(hookB)

                # guess feature num
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    break
                assert featureA[0].shape == featureB[0].shape
                num_features = featureA[0].view(featureA[0].size(0), -1).shape[-1]
                featureA = []
                featureB = []

                features = np.zeros((self.train_data_num, num_features * 2), dtype=np.float32)
                labels = np.zeros(self.train_data_num, dtype=np.int64)

                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    features[index] = np.concatenate((featureA[i].view(featureA[i].size(0), -1).numpy(),
                                        featureB[i].view(featureB[i].size(0), -1).numpy()),
                                       axis=1)
                    labels[index] = target
                    print("\rget clustering data: [{}/{}]".format(i+1, self.train_batch_num), end='')
                handleA.remove()
                handleB.remove()
                print("\nFinish collect cluster data.")

        elif mode == 'single':
            featureA = []

            def hookA(module, input, output):
                featureA.append(output.clone().cpu().detach())

            self.modelA.eval()
            with torch.no_grad():
                if 'convnext' in self.args.backbone.lower():
                    handleA = self.modelA.head.flatten.register_forward_hook(hookA)
                else:
                    layer_num = 0
                    for i in self.modelA.modules():
                        layer_num += 1
                    target_layer_ind = layer_num - 2
                    for i, j in enumerate(self.modelA.modules()):
                        if i == target_layer_ind:
                            handleA = j.register_forward_hook(hookA)

                # guess feature num
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    break
                num_features = featureA[0].view(featureA[0].size(0), -1).shape[-1]
                featureA = []

                features = np.zeros((self.train_data_num, num_features), dtype=np.float32)
                labels = np.zeros(self.train_data_num, dtype=np.int64)

                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    features[index] = featureA[i].view(featureA[i].size(0), -1).numpy()
                    labels[index] = target
                    print("\rget clustering data: [{}/{}]".format(i + 1, self.train_batch_num), end='')
                handleA.remove()
                print("\nFinish collect cluster data.")

        elif mode == 'dual_PCA':
            featureA = []
            def hookA(module, input, output):
                featureA.append(output.clone().cpu().detach())

            featureB = []
            def hookB(module, input, output):
                featureB.append(output.clone().cpu().detach())

            self.modelA.eval()
            with torch.no_grad():
                if 'convnext' in self.args.backbone.lower():
                    handleA = self.modelA.head.flatten.register_forward_hook(hookA)
                    handleB = self.modelB.head.flatten.register_forward_hook(hookB)
                else:
                    layer_num = 0
                    for i in self.modelA.modules():
                        layer_num += 1
                    target_layer_ind = layer_num - 2
                    for i, j in enumerate(self.modelA.modules()):
                        if i == target_layer_ind:
                            handleA = j.register_forward_hook(hookA)
                    for i, j in enumerate(self.modelB.modules()):
                        if i == target_layer_ind:
                            handleB = j.register_forward_hook(hookB)

                # guess feature num
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    break
                assert featureA[0].shape == featureB[0].shape
                num_features = featureA[0].view(featureA[0].size(0), -1).shape[-1]
                featureA = []
                featureB = []

                features = np.zeros((self.train_data_num, self.args.dim_reduce), dtype=np.float32)
                labels = np.zeros(self.train_data_num, dtype=np.int64)

                featureA_ = np.zeros((self.train_data_num, num_features), dtype=np.float32)
                featureB_ = np.zeros((self.train_data_num, num_features), dtype=np.float32)
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    featureA_[index] = featureA[i].view(featureA[i].size(0), -1).numpy()
                    featureB_[index] = featureB[i].view(featureB[i].size(0), -1).numpy()
                    labels[index] = target
                    print("\rget clustering data: [{}/{}]".format(i+1, self.train_batch_num), end='')

                target_features = self.args.dim_reduce//2

                pca = PCA(n_components=target_features, copy=False)
                featureA = pca.fit_transform(np.array(featureA_))
                featureB = pca.fit_transform(np.array(featureB_))

                features[:, target_features:] = featureA
                features[:, :target_features] = featureB

                handleA.remove()
                handleB.remove()
                print("\nFinish collect cluster data.")
        else:
            NotImplementedError("mode {} not been implemneted!!!".format(mode))

        return features, labels

    def _cluster_data_into_subsets(self):
        features, labels = self._gen_clustering_data(self.args.cluster_mode)
        cc = CurriculumClustering(n_subsets=3, verbose=True, random_state=0, dim_reduce=self.args.dim_reduce)
        cc.fit(features, labels)
        self.subset_labels = cc.output_labels
        np.save(join(self.args.dir, 'subset_labels.npy'), self.subset_labels)
        # 특징 데이터 저장 (t-SNE 시각화용)
        np.save(join(self.args.dir, 'clustering_result.npy'), features)

    def _get_label_update_stage(self, epoch):
        update_stage = 0
        step = (self.args.stage2 - self.args.stage1) / 3
        for i in range(3):
            if self.args.stage1 + step * (i + 1) < epoch:
                update_stage += 1
        return update_stage

    def training(self):
        # 전체 학습 루프 실행 함수
        timer = AverageMeter()
        # train
        end = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            print('-----------------')
            
            # unfreeze
            if epoch == self.args.warmup:
                print(f"Epoch {epoch} | Backbone Unfreeze")
                for name, param in self.modelA.named_parameters():
                    if "head" not in name: param.requires_grad = True
                for name, param in self.modelB.named_parameters():
                    if "head" not in name: param.requires_grad = True

                if self.args.optim == 'ASAM':
                    rho = getattr(self.args, 'rho', 0.5)
                    eta = getattr(self.args, 'eta', 0.01)
                    
                    self.optimizerA = self._get_optim(self.modelA.parameters(), optim='SGD')
                    self.optimizerB = self._get_optim(self.modelB.parameters(), optim='SGD')
                    
                    self.optimizerA = ASAM(self.optimizerA, self.modelA, rho=rho, eta=eta)
                    self.optimizerB = ASAM(self.optimizerB, self.modelB, rho=rho, eta=eta)
                else:
                    self.optimizerA = self._get_optim(self.modelA.parameters(), optim=self.args.optim)
                    self.optimizerB = self._get_optim(self.modelB.parameters(), optim=self.args.optim)

                for pg in self.optimizerA.param_groups: pg['lr'] = self.args.lr2
                for pg in self.optimizerB.param_groups: pg['lr'] = self.args.lr2

                if self.args.scheduler == "Cosine":
                    remain = self.args.epochs - epoch
                    self.schedulerA = CosineAnnealingLR(self.optimizerA, T_max=remain, eta_min=1e-6)
                    self.schedulerB = CosineAnnealingLR(self.optimizerB, T_max=remain, eta_min=1e-6)
            
            if self.args.warmup > 0 and epoch < self.args.warmup:
                drop_path_rate = 0.0
            else:
                drop_path_rate = self.args.drop_path_rate

            self._update_drop_path(self.modelA, drop_path_rate)
            self._update_drop_path(self.modelB, drop_path_rate)

            if self.args.scheduler == "Cosine":
                if hasattr(self, 'schedulerA'): self.schedulerA.step()
                if hasattr(self, 'schedulerB'): self.schedulerB.step()
            else:
                self._adjust_learning_rate(epoch)

            # load y_tilde
            if os.path.isfile(self.args.y_file):
                self.yy = np.load(self.args.y_file)
            else:
                self.yy = []

            if epoch == self.args.stage1:
                self._cluster_data_into_subsets()

            if self.args.classnum > 5:
                train_prec1_A, train_prec1_B, train_prec5_A, train_prec5_B = self.train(epoch)
                val_prec1_A, val_prec1_B, val_prec5_A, val_prec5_B = self.val(epoch)
                test_prec1_A, test_prec1_B, test_prec5_A, test_prec5_B = self.test(epoch)
            else:
                train_prec1_A, train_prec1_B = self.train(epoch)
                val_prec1_A, val_prec1_B = self.val(epoch)
                test_prec1_A, test_prec1_B = self.test(epoch)

            # load best model to modelC
            if epoch < self.args.stage1:
                best_ind = [val_prec1_A, val_prec1_B, self.best_prec1].index(max(val_prec1_A, val_prec1_B, self.best_prec1))
                self.modelC.load_state_dict([self.modelA.state_dict(), self.modelB.state_dict(), self.modelC.state_dict()][best_ind])

            is_best = max(val_prec1_A, val_prec1_B) > self.best_prec1
            self.best_prec1 = max(val_prec1_A, val_prec1_B, self.best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.backbone,
                'state_dict_A': self.modelA.state_dict(),
                'optimizer_A': self.optimizerA.state_dict(),
                'state_dict_B': self.modelB.state_dict(),
                'optimizer_B': self.optimizerB.state_dict(),
                'prec_A': val_prec1_A,
                'prec_B': val_prec1_B,
                'best_prec1': self.best_prec1,
            }, is_best, filename=self.args.checkpoint_dir, modelbest=self.args.modelbest_dir)

            self._record()

            timer.update(time.time() - end)
            end = time.time()
            print("Epoch {} using {} min {:.2f} sec".format(epoch, timer.val // 60, timer.val % 60))


    # Train Loop
    def train(self, epoch=0):
        # 1 에폭(Epoch) 동안의 학습 수행
        batch_time = AverageMeter()
        losses_A = AverageMeter()
        losses_B = AverageMeter()
        top1_A = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()
        pure_ratio_A = AverageMeter()
        pure_ratio_B = AverageMeter()
        pure_ratio_discard_A = AverageMeter()
        pure_ratio_discard_B = AverageMeter()
        margin_accu = AverageMeter()
        if self.args.classnum > 5:
            top5_A = AverageMeter()
            top5_B = AverageMeter()
            top5_mix = AverageMeter()
        end = time.time()

        self.modelA.train()
        self.modelB.train()

        # new y is y_tilde after updating
        self.new_y = np.zeros([self.args.datanum, self.args.classnum])
        if epoch >= self.args.stage1:
            self.new_y = self.yy

        for i, (input, target, index) in enumerate(self.trainloader):

            index = index.numpy()

            input = input.to(self.args.device)
            target1 = target.to(self.args.device).long()
            input_var = input.clone().to(self.args.device)
            target_var = target1.clone().to(self.args.device)

            outputA = self.modelA(input_var)
            outputB = self.modelB(input_var)
            output_mix = (outputA + outputB) / 2

            if epoch < self.args.warmup:
                
                lossA, lossB, yy_A, yy_B, ind_A_discard, ind_B_discard, ind_A_update, ind_B_update, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self._compute_loss(outputA, outputB, target1, target_var,
                                                                                index, epoch, i)
            else:
                lossA, lossB, yy_A, yy_B, ind_A_discard, ind_B_discard, ind_A_update, ind_B_update, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self._compute_loss(outputA, outputB, target, target_var,
                                                                                index, epoch, i)

            outputA_ = outputA
            outputB_ = outputB

            outputA = F.softmax(outputA, dim=1)
            outputB = F.softmax(outputB, dim=1)
            output_mix = F.softmax(output_mix, dim=1)

            # Update recorder
            if self.args.classnum > 5:
                prec1_A, prec5_A = accuracy(outputA.data, target1, topk=(1,5))
                prec1_B, prec5_B = accuracy(outputB.data, target1, topk=(1,5))
                prec1_mix, prec5_mix = accuracy(output_mix.data, target1, topk=(1,5))
            else:
                prec1_A = accuracy(outputA.data, target1, topk=(1,))
                prec1_B = accuracy(outputB.data, target1, topk=(1,))
                prec1_mix = accuracy(output_mix.data, target1, topk=(1,))
            top1_A.update(float(prec1_A[0]), input.shape[0])
            top1_B.update(float(prec1_B[0]), input.shape[0])
            top1_mix.update(float(prec1_mix[0]), input.shape[0])
            losses_A.update(float(lossA.data))
            losses_B.update(float(lossB.data))
            pure_ratio_A.update(pure_ratio_1)
            pure_ratio_B.update(pure_ratio_2)
            if pure_ratio_discard_1 >= 0 and pure_ratio_discard_2 >= 0:
                pure_ratio_discard_A.update(pure_ratio_discard_1)
                pure_ratio_discard_B.update(pure_ratio_discard_2)
            if self.args.classnum > 5:
                top5_A.update(float(prec5_A), input.shape[0])
                top5_B.update(float(prec5_B), input.shape[0])
                top5_mix.update(float(prec5_mix), input.shape[0])

            # --- ASAM Optimizer Step ---
            # ASAM이 활성화된 경우 Ascent -> Descent 과정을 수행
            if self.args.optim == 'ASAM' and epoch >= self.args.warmup: 
                # last_y_var_A/B 정의 (ASAM 타겟으로 사용됨)
                if epoch >= self.args.stage1:
                    last_y_var_A = self.softmax(yy_A)
                    last_y_var_B = self.softmax(yy_B)
                
                # 1. Update Model A
                self.optimizerA.zero_grad()
                lossA.backward() 
                self.optimizerA.ascent_step()

                # Re-compute Loss A (Adv)
                # Ascent된 가중치로 다시 Forward
                outputA_adv = self.modelA(input_var)
                
                # Loss 재계산: Selection은 이전 단계의 indices 재사용 (ind_B_update: B가 선택한 샘플로 A 학습)
            
                # Stage별 Loss Type 결정
                if epoch < self.args.stage1:
                    l_type = self.args.cost_type
                    # Stage 1: Target is noisy label (target_var)
                    adv_lossA = self._get_loss(outputA_adv[ind_B_update], target_var[ind_B_update], loss_type=l_type, beta=self.args.beta)
                elif epoch < self.args.stage2:
                    # Stage 2: Target is Soft Label (yy_A -> last_y_var_A)
                    # last_y_var_A는 Softmax(yy_A)임.
                    adv_lossA = self._get_loss(outputA_adv[ind_B_update], last_y_var_A[ind_B_update], 
                                               alpha=self.args.alpha, beta=self.args.beta, loss_type='PENCIL', target_var=target_var[ind_B_update])
                else:
                    # Stage 3: Fine-tuning
                    adv_lossA = self._get_loss(outputA_adv[ind_B_update], last_y_var_A[ind_B_update], loss_type='PENCIL_KL')
                
                # adv_lossA는 mean reduction 되어 있음 (get_loss 기본값)
                # backward
                adv_lossA.backward()
                self.optimizerA.descent_step()
                
                # 2. Update Model B
                self.optimizerB.zero_grad()
                lossB.backward() 
                self.optimizerB.ascent_step()
                
                # Re-compute Loss B (Adv)
                outputB_adv = self.modelB(input_var)
                
                if epoch < self.args.stage1:
                    l_type = self.args.cost_type
                    adv_lossB = self._get_loss(outputB_adv[ind_A_update], target_var[ind_A_update], loss_type=l_type, beta=self.args.beta)
                elif epoch < self.args.stage2:
                    adv_lossB = self._get_loss(outputB_adv[ind_A_update], last_y_var_B[ind_A_update], 
                                               alpha=self.args.alpha, beta=self.args.beta, loss_type='PENCIL', target_var=target_var[ind_A_update])
                else:
                    adv_lossB = self._get_loss(outputB_adv[ind_A_update], last_y_var_B[ind_A_update], loss_type='PENCIL_KL')

                adv_lossB.backward()
                self.optimizerB.descent_step()

            else:
                # Normal SGD Step
                self.optimizerA.zero_grad()
                lossA.backward()
                self.optimizerA.step()

                self.optimizerB.zero_grad()
                lossB.backward()
                self.optimizerB.step()

            # update label distribution
            if epoch >= self.args.stage1 and epoch < self.args.stage2:
                # using select data sample update parameters, other update label only
                yy_A = self.yy
                yy_B = self.yy
                yy_A = torch.tensor(yy_A[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
                yy_B = torch.tensor(yy_B[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
                # obtain label distributions (y_hat)
                last_y_var_A = self.softmax(yy_A)
                last_y_var_B = self.softmax(yy_B)
                lossA = self._get_loss(outputA_.detach(), last_y_var_A, loss_type="PENCIL", target_var=target_var)
                lossB = self._get_loss(outputB_.detach(), last_y_var_B, loss_type="PENCIL", target_var=target_var)

                lossA.backward()
                lossB.backward()

                grad = yy_A.grad.data + yy_B.grad.data
                if self.args.mix_grad == 1:
                    yy_A.data.sub_(self.args.lambda1 * grad)
                    yy_B.data.sub_(self.args.lambda1 * grad)
                else:
                    yy_A.data.sub_(self.args.lambda1 * yy_A.grad.data)
                    yy_B.data.sub_(self.args.lambda1 * yy_B.grad.data)

                if self.args.discard == 1:
                    if self.args.curriculum == 0:
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        self.new_y[index[ind_discard], :] = yy_A.data[ind_discard].cpu().numpy()
                    else:
                        # choose update labels
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        update_stage = self._get_label_update_stage(epoch)
                        subset_labels = self.subset_labels[index[ind_discard]]
                        update_ind = np.where(subset_labels <= update_stage)
                        self.new_y[index[ind_discard[update_ind]], :] = yy_A.data[ind_discard[update_ind]].cpu().numpy()
                else:
                    if self.args.curriculum == 0:
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        self.new_y[index, :] = yy_A.data.cpu().numpy()
                    else:
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        update_stage = self._get_label_update_stage(epoch)
                        subset_labels = self.subset_labels[index]
                        update_ind = np.where(subset_labels <= update_stage)
                        self.new_y[index[update_ind], :] = yy_A.data[update_ind].cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print("\rTrain Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                  "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                  "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                  "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                  "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                epoch, self.args.epochs, i, self.train_batch_num,
                batch_time=batch_time, loss_A=losses_A, loss_B=losses_B, top1_A=top1_A, top1_B=top1_B), end='')

        if epoch < self.args.stage2:
            # save y_tilde
            self.yy = self.new_y
            y_file = join(self.args.dir, "y.npy")
            np.save(y_file, self.new_y)
            y_record = join(self.args.dir, "record/y_%03d.npy" % epoch)
            np.save(y_record, self.new_y)

        # check label acc
        label_accu_A, label_n2t_A, label_t2n_A = check_label(self.yy, self.clean_labels, self.noise_or_not, onehotA=True)
        label_accu_B, label_n2t_B, label_t2n_B = check_label(self.yy, self.clean_labels, self.noise_or_not, onehotA=True)
        label_accu_C, label_n2t_C, label_t2n_C = check_label(self.yy, self.clean_labels, self.noise_or_not, onehotA=True)

        if self.args.classnum > 5:
            print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}\tTop5 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg,
                                                                                                 top5_A.avg,
                                                                                                 top5_B.avg))
        else:
            print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))
        print(" * Label accu A: {:.4f}\tB: {:.4f}\tPure ratio A: {:.4f}\tB: {:.4f}".format(label_accu_A, label_accu_B,
                                                                                           pure_ratio_A.avg,
                                                                                           pure_ratio_B.avg))
        print(" * n2t_A: {:.4f}\tn2t_B: {:.4f}\tt2n_A: {:.4f}\tt2n_B: {:.4f}".format(label_n2t_A, label_n2t_B,
                                                                                     label_t2n_A, label_t2n_B))

        self.record_dict['train1']['acc'].append(top1_A.avg)
        self.record_dict['train1']['loss'].append(losses_A.avg)
        self.record_dict['train1']['label_n2t'].append(label_n2t_A)
        self.record_dict['train1']['label_t2n'].append(label_t2n_A)
        self.record_dict['train1']['label_accu'].append(label_accu_A)
        self.record_dict['train1']['pure_ratio'].append(pure_ratio_A.avg)
        self.record_dict['train1']['pure_ratio_discard'].append(pure_ratio_discard_A.avg)
        self.record_dict['train1']['margin_accu'].append(margin_accu.avg)

        self.record_dict['train2']['acc'].append(top1_B.avg)
        self.record_dict['train2']['loss'].append(losses_B.avg)
        self.record_dict['train2']['label_n2t'].append(label_n2t_B)
        self.record_dict['train2']['label_t2n'].append(label_t2n_B)
        self.record_dict['train2']['label_accu'].append(label_accu_B)
        self.record_dict['train2']['pure_ratio'].append(pure_ratio_B.avg)
        self.record_dict['train2']['pure_ratio_discard'].append(pure_ratio_discard_B.avg)
        self.record_dict['train2']['margin_accu'].append(margin_accu.avg)

        self.record_dict['train3']['acc'].append(top1_mix.avg)
        self.record_dict['train3']['label_n2t'].append(label_n2t_C)
        self.record_dict['train3']['label_t2n'].append(label_t2n_C)
        self.record_dict['train3']['label_accu'].append(label_accu_C)
        if self.args.classnum > 5:
            self.record_dict['train1']['acc5'].append(top5_A.avg)
            self.record_dict['train2']['acc5'].append(top5_B.avg)
            self.record_dict['train3']['acc5'].append(top5_mix.avg)
            return top1_A.avg, top1_B.avg, top5_A.avg, top5_B.avg
        return top1_A.avg, top1_B.avg


    def val(self, epoch=0):
        self.modelA.eval()
        self.modelB.eval()

        batch_time = AverageMeter()
        losses_A = AverageMeter()
        top1_A = AverageMeter()
        losses_B = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()
        if self.args.classnum > 5:
            top5_A = AverageMeter()
            top5_B = AverageMeter()
            top5_mix = AverageMeter()

        with torch.no_grad():
            # Validate
            end = time.time()
            for i, (img, label, index) in enumerate(self.valloader):

                img = img.to(self.args.device)
                label = label.to(self.args.device).long()

                outputA = self.modelA(img)
                lossA = self.criterion(outputA, label)
                outputB = self.modelB(img)
                lossB = self.criterion(outputB, label)
                output_mix = (outputA + outputB) / 2

                outputA = F.softmax(outputA, dim=1)
                outputB = F.softmax(outputB, dim=1)
                output_mix = F.softmax(output_mix, dim=1)

                # Update recorder
                if self.args.classnum > 5:
                    prec1_A, prec5_A = accuracy(outputA.data, label, topk=(1, 5))
                    prec1_B, prec5_B = accuracy(outputB.data, label, topk=(1, 5))
                    prec1_mix, prec5_mix = accuracy(output_mix.data, label, topk=(1, 5))
                else:
                    prec1_A = accuracy(outputA.data, label, topk=(1,))
                    prec1_B = accuracy(outputB.data, label, topk=(1,))
                    prec1_mix = accuracy(output_mix.data, label, topk=(1,))
                top1_A.update(float(prec1_A[0]), img.shape[0])
                losses_A.update(float(lossA.data))
                top1_B.update(float(prec1_B[0]), img.shape[0])
                losses_B.update(float(lossB.data))
                top1_mix.update(float(prec1_mix[0]), img.shape[0])
                if self.args.classnum > 5:
                    top5_A.update(float(prec5_A), img.shape[0])
                    top5_B.update(float(prec5_B), img.shape[0])
                    top5_mix.update(float(prec5_mix), img.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print("\rVal Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  " 
                      "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                      "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                      "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                      "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                    epoch, self.args.epochs, i, self.val_batch_num,
                    batch_time=batch_time, loss_A=losses_A, loss_B=losses_B, top1_A=top1_A, top1_B=top1_B), end='')

            if self.args.classnum > 5:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}\tTop5 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg,
                                                                                                   top5_A.avg, top5_B.avg))
            else:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))

        self.record_dict['val1']['acc'].append(top1_A.avg)
        if self.args.classnum > 5:
            self.record_dict['val1']['acc5'].append(top5_A.avg)
        self.record_dict['val1']['loss'].append(losses_A.avg)

        self.record_dict['val2']['acc'].append(top1_B.avg)
        if self.args.classnum > 5:
            self.record_dict['val2']['acc5'].append(top5_B.avg)
        self.record_dict['val2']['loss'].append(losses_B.avg)

        self.record_dict['val3']['acc'].append(top1_mix.avg)
        if self.args.classnum > 5:
            self.record_dict['val3']['acc5'].append(top5_mix.avg)

        if self.args.classnum > 5:
            return top1_A.avg, top1_B.avg, top5_A.avg, top5_B.avg
        return top1_A.avg, top1_B.avg

    def test(self, epoch=0):
        self.modelA.eval()
        self.modelB.eval()

        batch_time = AverageMeter()
        losses_A = AverageMeter()
        top1_A = AverageMeter()
        losses_B = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()
        if self.args.classnum > 5:
            top5_A = AverageMeter()
            top5_B = AverageMeter()
            top5_mix = AverageMeter()

        with torch.no_grad():
            # Validate
            end = time.time()
            for i, (img, label, index) in enumerate(self.testloader):

                img = img.to(self.args.device)
                label = label.to(self.args.device).long()

                outputA = self.modelA(img)
                lossA = self.criterion(outputA, label)
                outputB = self.modelB(img)
                lossB = self.criterion(outputB, label)
                output_mix = (outputA + outputB) / 2

                outputA = F.softmax(outputA, dim=1)
                outputB = F.softmax(outputB, dim=1)
                output_mix = F.softmax(output_mix, dim=1)

                # Update recorder
                if self.args.classnum > 5:
                    prec1_A, prec5_A = accuracy(outputA.data, label, topk=(1, 5))
                    prec1_B, prec5_B = accuracy(outputB.data, label, topk=(1, 5))
                    prec1_mix, prec5_mix = accuracy(output_mix.data, label, topk=(1, 5))
                else:
                    prec1_A = accuracy(outputA.data, label, topk=(1,))
                    prec1_B = accuracy(outputB.data, label, topk=(1,))
                    prec1_mix = accuracy(output_mix.data, label, topk=(1,))
                top1_A.update(float(prec1_A[0]), img.shape[0])
                losses_A.update(float(lossA.data))
                top1_B.update(float(prec1_B[0]), img.shape[0])
                losses_B.update(float(lossB.data))
                top1_mix.update(float(prec1_mix[0]), img.shape[0])
                if self.args.classnum > 5:
                    top5_A.update(float(prec5_A), img.shape[0])
                    top5_B.update(float(prec5_B), img.shape[0])
                    top5_mix.update(float(prec5_mix), img.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print("\rTest Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                      "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                      "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                      "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                      "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                    epoch, self.args.epochs, i, self.test_batch_num,
                    batch_time=batch_time, loss_A=losses_A, loss_B=losses_B, top1_A=top1_A, top1_B=top1_B), end='')

            if self.args.classnum > 5:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}\tTop5 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg,
                                                                                                   top5_A.avg, top5_B.avg))
            else:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))

        self.record_dict['test1']['acc'].append(top1_A.avg)
        if self.args.classnum > 5:
            self.record_dict['test1']['acc5'].append(top5_A.avg)
        self.record_dict['test1']['loss'].append(losses_A.avg)

        self.record_dict['test2']['acc'].append(top1_B.avg)
        if self.args.classnum > 5:
            self.record_dict['test2']['acc5'].append(top5_B.avg)
        self.record_dict['test2']['loss'].append(losses_B.avg)

        self.record_dict['test3']['acc'].append(top1_mix.avg)
        if self.args.classnum > 5:
            self.record_dict['test3']['acc5'].append(top5_mix.avg)

        if self.args.classnum > 5:
            return top1_A.avg, top1_B.avg, top5_A.avg, top5_B.avg
        return top1_A.avg, top1_B.avg


def save_checkpoint(state, is_best, filename='', modelbest=''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)
        print("Saving best model at epoch {}".format(state['epoch']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 원래는 reshape가 아니라 view였음
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    args = get_args()
    trainer = CoCorrecting(args=args)
    trainer.training()