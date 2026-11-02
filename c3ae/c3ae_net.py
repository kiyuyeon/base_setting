#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
import random
from PIL import Image
import pandas as pd
from pathlib import Path
import copy
from sklearn.metrics import f1_score, confusion_matrix

# ========================================
# 1. C3AE Model Definition
# ========================================
class C3AE(nn.Module):
    def __init__(self, feat_dim=12):
        super().__init__()
        # 3-scale 입력을 채널축으로 합치므로 in_ch=9 (3*RGB)
        self.feature = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(2,2),

            nn.Conv2d(32, 32, 3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(2,2),

            nn.Conv2d(32, 32, 3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(2,2),

            nn.Conv2d(32, 32, 3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1)
        )
        # 최종: (B,32,4,4)
        self.f1 = nn.Linear(32*4*4, feat_dim)
        self.f2 = nn.Linear(feat_dim, 1)

    def forward(self, x9ch):
        # x9ch: (B,9,64,64)
        feat_map = self.feature(x9ch)              # (B,32,4,4)
        feat = feat_map.flatten(1)                 # (B,32*4*4)
        feat_logits = self.f1(feat)                # (B,feat_dim)
        pred_age = self.f2(feat_logits).squeeze(1) # (B,)
        return feat_logits, pred_age

# ========================================
# 2. MegaAge Dataset Class
# ========================================
class MegaAgeDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, name_file, age_file, new_size=(64, 64), augment=False):
        """
        MegaAge Asian 데이터셋을 위한 Dataset 클래스
        
        Args:
            image_dir: 이미지들이 있는 디렉토리 (train 또는 test)
            name_file: 이미지 이름이 적힌 파일 (train_name.txt, test_name.txt)
            age_file: 나이 정보가 적힌 파일 (train_age.txt, test_age.txt)
            new_size: 이미지 리사이즈 크기
            augment: 데이터 증강 여부
        """
        self.image_dir = image_dir
        self.new_size = new_size
        self.augment = augment
        
        # Load image names
        with open(name_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        
        # Load ages
        with open(age_file, 'r') as f:
            self.ages = [float(line.strip()) for line in f.readlines()]
        
        # 데이터 길이 확인
        assert len(self.image_names) == len(self.ages), \
            f"Mismatch: {len(self.image_names)} images vs {len(self.ages)} ages"
        
        print(f"Loaded {len(self.image_names)} samples from {image_dir}")
        
        # Augmentation transforms
        self.aug_transforms = None
        if augment:
            self.aug_transforms = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
            ])
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Get image name and age
        img_name = self.image_names[idx]
        age = self.ages[idx]
        
        # Clamp age to reasonable range
        age = max(0, min(100, age))
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            # Fallback to random valid image
            print(f"Warning: Could not load {img_path}, using fallback")
            fallback_idx = random.randint(0, len(self.image_names) - 1)
            return self.__getitem__(fallback_idx)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        
        # Apply augmentation if specified
        if self.aug_transforms is not None:
            image = self.aug_transforms(image)
        
        # Create 3-scale version and process
        x = self.create_multiscale_input(image)
        
        # Age group (0: <10, 1:10-19, ..., 7:70+)
        age_group = min(int(age // 10), 7)
        
        return x, float(age), age_group
    
    def create_multiscale_input(self, pil_image):
        """PIL 이미지를 3-scale 9채널 텐서로 변환"""
        # Create 3 different scales
        img64 = pil_image.resize(self.new_size, Image.BILINEAR)
        img32 = pil_image.resize((self.new_size[0]//2, self.new_size[1]//2), Image.BILINEAR)
        img16 = pil_image.resize((self.new_size[0]//4, self.new_size[1]//4), Image.BILINEAR)
        
        # Convert to tensors and normalize
        to_tensor = T.Compose([
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1]
        ])
        
        t64 = to_tensor(img64)  # (3, 64, 64)
        t32 = to_tensor(img32)  # (3, 32, 32)
        t16 = to_tensor(img16)  # (3, 16, 16)
        
        # Upsample smaller scales to 64x64
        t32_up = F.interpolate(t32.unsqueeze(0), size=self.new_size, 
                              mode='bilinear', align_corners=False).squeeze(0)
        t16_up = F.interpolate(t16.unsqueeze(0), size=self.new_size, 
                              mode='bilinear', align_corners=False).squeeze(0)
        
        # Concatenate along channel dimension: (9, 64, 64)
        x = torch.cat([t64, t32_up, t16_up], dim=0)
        
        return x

# ========================================
# 3. Training Utilities
# ========================================
def gaussian_soft_labels(y_grp, num_bins=12, sigma=1.0):
    """Gaussian soft labeling"""
    B = y_grp.shape[0]
    bins = torch.arange(num_bins, device=y_grp.device).float()
    y = y_grp.view(-1, 1).float()
    dist2 = (bins.view(1, -1) - y) ** 2
    w = torch.exp(-0.5 * dist2 / (sigma ** 2))
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    return w

@torch.no_grad()
def age_to_group(age):
    """Age를 age group으로 변환"""
    g = torch.full_like(age, 7, dtype=torch.long)
    g = torch.where(age < 70, torch.tensor(6, device=age.device), g)
    g = torch.where(age < 60, torch.tensor(5, device=age.device), g)
    g = torch.where(age < 50, torch.tensor(4, device=age.device), g)
    g = torch.where(age < 40, torch.tensor(3, device=age.device), g)
    g = torch.where(age < 30, torch.tensor(2, device=age.device), g)
    g = torch.where(age < 20, torch.tensor(1, device=age.device), g)
    g = torch.where(age < 10, torch.tensor(0, device=age.device), g)
    return g

# ========================================
# 4. Training Functions
# ========================================
def train_one_epoch(model, loader, optimizer, device, feat_bins=12, lambda_kl=1e-5, sigma=1.0):
    model.train()
    l_sum = n = 0
    
    for batch_idx, (x, y_age, y_grp) in enumerate(loader):
        x = x.to(device)
        y_age = y_age.to(device)
        y_grp = y_grp.to(device)
        
        feat, pred = model(x)  # feat:(B,feat_bins), pred:(B,)
        
        # L1 loss for age regression
        l1 = F.l1_loss(pred, y_age)
        
        # KL divergence loss with soft labels
        y_soft = gaussian_soft_labels(y_grp, feat_bins, sigma=sigma)
        logp = F.log_softmax(feat, dim=1)
        kl = F.kl_div(logp, y_soft, reduction='batchmean')
        
        # L1 regularization on f1 weights
        lreg = 0.0
        for p in model.f1.parameters():
            lreg = lreg + p.abs().sum()
        
        loss = l1 + lambda_kl * (kl + 1e-4 * lreg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        bs = x.size(0)
        l_sum += loss.item() * bs
        n += bs
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(loader)}: Loss = {loss.item():.4f}, L1 = {l1.item():.4f}, KL = {kl.item():.4f}')
    
    return l_sum / max(1, n)

@torch.no_grad()
def evaluate_f1(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    age_errors = []
    
    for x, y_age, y_grp in loader:
        x = x.to(device)
        y_age = y_age.to(device)
        
        _, pred = model(x)
        g = age_to_group(pred.clamp(min=0, max=100))
        
        y_true.extend(y_grp.cpu().numpy().tolist())
        y_pred.extend(g.cpu().numpy().tolist())
        age_errors.extend(torch.abs(pred - y_age).cpu().numpy().tolist())
    
    f1w = f1_score(y_true, y_pred, average='weighted')
    f1m = f1_score(y_true, y_pred, average='macro')
    mae = np.mean(age_errors)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))
    
    return f1w, f1m, mae, cm

# ========================================
# 5. Main Training Function
# ========================================
def train_megaage_c3ae(data_root, epochs=50, batch_size=32, lr=1e-3, 
                       feat_bins=12, device=None, save_dir='./checkpoints'):
    """
    MegaAge Asian 데이터셋으로 C3AE 모델 학습
    
    Args:
        data_root: 데이터 루트 디렉토리 (megaage_asian 폴더)
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        lr: 학습률
        feat_bins: 특징 차원
        device: 디바이스
        save_dir: 체크포인트 저장 디렉토리
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Data paths
    train_img_dir = os.path.join(data_root, 'train')
    test_img_dir = os.path.join(data_root, 'test')
    list_dir = os.path.join(data_root, 'list')
    
    train_name_file = os.path.join(list_dir, 'train_name.txt')
    train_age_file = os.path.join(list_dir, 'train_age.txt')
    test_name_file = os.path.join(list_dir, 'test_name.txt')
    test_age_file = os.path.join(list_dir, 'test_age.txt')
    
    # Check if files exist
    for f in [train_name_file, train_age_file, test_name_file, test_age_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")
    
    # Datasets and DataLoaders
    train_dataset = MegaAgeDataset(train_img_dir, train_name_file, train_age_file, 
                                  new_size=(64, 64), augment=True)
    test_dataset = MegaAgeDataset(test_img_dir, test_name_file, test_age_file, 
                                 new_size=(64, 64), augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Model
    model = C3AE(feat_dim=feat_bins).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.1)
    
    # Training loop
    best_f1 = -1
    best_mae = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, 
                                   feat_bins=feat_bins, lambda_kl=1e-5, sigma=1.0)
        
        # Validate
        f1w, f1m, mae, cm = evaluate_f1(model, test_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test F1(weighted): {f1w:.4f} | F1(macro): {f1m:.4f}")
        print(f"Test MAE: {mae:.2f} | LR: {current_lr:.6f}")
        
        # Save best model
        if f1w > best_f1:
            best_f1 = f1w
            best_mae = mae
            best_weights = copy.deepcopy(model.state_dict())
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'f1_weighted': f1w,
                'f1_macro': f1m,
                'mae': mae,
                'confusion_matrix': cm.tolist()
            }
            torch.save(checkpoint, os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth'))
            print(f"★ New best model saved! F1: {f1w:.4f}, MAE: {mae:.2f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Load best weights
    model.load_state_dict(best_weights)
    
    # Final evaluation
    print(f"\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    final_f1w, final_f1m, final_mae, final_cm = evaluate_f1(model, test_loader, device)
    print(f"Best F1(weighted): {final_f1w:.4f}")
    print(f"Best F1(macro): {final_f1m:.4f}")
    print(f"Best MAE: {final_mae:.2f}")
    print(f"Confusion Matrix:")
    print("Age groups: 0:<10, 1:10-19, 2:20-29, 3:30-39, 4:40-49, 5:50-59, 6:60-69, 7:70+")
    print(final_cm)
    
    return model, (final_f1w, final_f1m, final_mae, final_cm)

# ========================================
# 6. Usage Example
# ========================================
if __name__ == "__main__":
    # 데이터 경로 설정 (실제 경로로 수정하세요)
    DATA_ROOT = r"C:\Users\chwo6\OneDrive\바탕 화면\cuda_test\megaage_asian"
    
    # GPU 메모리에 따라 배치 크기 조정 (GTX 1060 6GB 기준으로 32 권장)
    BATCH_SIZE = 32  # GPU 메모리가 부족하면 16으로 줄이세요
    
    # 학습 실행
    print("Starting C3AE training on MegaAge Asian dataset...")
    print(f"Data root: {DATA_ROOT}")
    
    model, metrics = train_megaage_c3ae(
        data_root=DATA_ROOT,
        epochs=50,              # 에폭 수
        batch_size=BATCH_SIZE,  # 배치 크기
        lr=1e-3,               # 학습률
        feat_bins=12,          # 특징 차원
        device=None,           # 자동으로 GPU/CPU 선택
        save_dir='./checkpoints'  # 체크포인트 저장 경로
    )
    
    print("Training completed successfully!")