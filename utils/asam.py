import torch
import torch.nn as nn
from collections import defaultdict

class ASAM(torch.optim.Optimizer):
    """
    ASAM (Adaptive Sharpness-Aware Minimization) Optimizer
    
    Improves SAM by adaptively adjusting the perturbation (epsilon) scale 
    according to the parameter scale.
    
    Reference: "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks" (ICML 2021)
    
    Mechanism:
    1. Ascent Step: Add perturbation (epsilon) in the direction of maximizing loss. 
       Epsilon is normalized proportional to element-wise parameter stats.
    2. Descent Step: Calculate gradient at perturbed weight (w + epsilon) 
       and update original weight (w).
    """
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        """
        Args:
            optimizer (torch.optim.Optimizer): Base optimizer (e.g. SGD). 
            model (nn.Module): Model to be optimized.
            rho (float): Neighborhood size (perturbation radius). Default 0.5.
            eta (float): Smoothing constant. Default 0.01.
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
        Move weights to maximize loss (Ascent) with element-wise normalization.
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

        # 3. Reset Gradients
        # Zero grads to prepare for re-calculation at w+epsilon
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        """
        Descent Step: 
        1. Keep gradients calculated at (w + epsilon).
        2. Restore original weights (w).
        3. Update w using gradients from step 1.
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

    @torch.no_grad()
    def step(self, closure=None):
        """
        Normal Step (e.g., for Warmup)
        Calls base optimizer step without ASAM logic.
        """
        return self.optimizer.step(closure)