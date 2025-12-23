import torch
import torch.nn as nn
from collections import defaultdict

class ASAM(torch.optim.Optimizer):
    """
    ASAM (Adaptive Sharpness-Aware Minimization) Optimizer
    
    이 클래스는 기존 SAM(Sharpness-Aware Minimization)을 개선하여, 
    파라미터의 스케일(Scale)에 따라 섭동(Epsilon)의 크기를 적응적으로(Adaptive) 조절하는 옵티마이저입니다.
    
    기반 논문: "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks" (ICML 2021)
    
    작동 원리:
    1. Ascent Step: 현재 가중치(w) 주변에서 Loss가 가장 가파르게 상승하는 방향으로 섭동(epsilon)을 더합니다. 
       이때 섭동의 크기는 파라미터의 요소별 크기(Element-wise stats)에 비례하여 정규화됩니다.
    2. Descent Step: 섭동된 가중치(w + epsilon) 위치에서의 그라디언트를 계산하고, 
       이를 원래 가중치(w)에 적용하여 업데이트합니다. (일반적인 SGD 업데이트)
    """
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        """
        초기화 함수
        
        Args:
            optimizer (torch.optim.Optimizer): 기본 옵티마이저 (예: SGD). 
                                             ASAM은 이 옵티마이저를 감싸서(Wrapping) 동작합니다.
            model (nn.Module): 학습할 모델. 파라미터 그룹 관리를 위해 필요합니다.
            rho (float): 섭동(Perturbation)의 반경(Neighborhood size). 기본값 0.5.
            eta (float): 수치 안정성을 위한 작은 상수. 기본값 0.01.
        """
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        
        # Ascent Step 이전의 원래 가중치(w) 상태를 저장하기 위한 딕셔너리
        self.state = defaultdict(dict)

        # 기본 옵티마이저의 초기 설정값을 상속 (defaults)
        super(ASAM, self).__init__(self.optimizer.param_groups, self.optimizer.defaults)

    @torch.no_grad()
    def ascent_step(self):
        """
        Ascent Step: w -> w + epsilon
        
        현재 그라디언트를 기반으로, Loss를 최대화하는 방향(Ascent)으로 가중치를 이동시킵니다.
        ASAM 논문의 핵심인 Element-wise Normalization이 여기서 적용됩니다.
        """
        # 1. Scale Factor 계산
        # 분모: || T_w \odot g ||_2
        # 여기서 T_w = |w| + eta
        norm_val = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                # T_w = |w| + eta
                t_w = torch.abs(p) + self.eta
                
                # T_w * g 의 제곱합을 누적
                weighted_grad = t_w * p.grad
                norm_val += torch.sum(weighted_grad ** 2)
        
        norm_val = torch.sqrt(norm_val) + 1e-12
        
        # 2. Perturbation 적용
        # Epsilon = rho * (T_w^2 * g) / || T_w * g ||
        scale = self.rho / norm_val
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                # 원래 상태 백업 (Descent Step에서 복구를 위해)
                self.state[p]["old_p"] = p.data.clone()
                
                # T_w = |w| + eta
                t_w = torch.abs(p) + self.eta
                
                # Epsilon 계산 및 적용
                # p.data += scale * (t_w ** 2) * p.grad
                epsilon = scale * (t_w ** 2) * p.grad
                p.data.add_(epsilon)

        # 3. 그라디언트 초기화 (다음 Backward를 위해)
        # Ascent Step 후 w+epsilon 상태에서 다시 Loss를 계산하고 Backward 해야 하므로,
        # 현재의 그라디언트(w에서의 기울기)는 지워줍니다.
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        """
        Descent Step: 
        1. 섭동된 가중치(w + epsilon)에서 계산된 그라디언트를 유지.
        2. 가중치를 원래 위치(w)로 복구.
        3. 복구된 w에 대해 1번의 그라디언트를 사용하여 SGD 업데이트 수행.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # 원래 가중치 복구
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        
        # 기본 옵티마이저의 step() 호출 (SGD Update)
        # 이때 p.grad는 "w + epsilon" 위치에서 계산된 그라디언트임.
        self.optimizer.step()
