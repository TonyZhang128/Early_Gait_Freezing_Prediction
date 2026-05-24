"""Gait self-supervised pre-training with SimCLR."""

import argparse
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from datasets.Contrastive_dataset import load_data
from models.resnet import ResNet18
from models.GSDNN_new import GSDNN_new
from models.Conformer import Conformer
from models.mambav3 import MambaV3
from utils.analysis import set_seed, save_checkpoint
from utils.calculate import FLOPs_calculat


# ==============================================================================
# Model & Loss
# ==============================================================================

class SimCLRModel(nn.Module):
    def __init__(self, base_model: nn.Module, out_dim: int = 32,
                 proj_out_dim: int = 128, contrastive_dim: int = 256,
                 dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout)
        self.projector = nn.Sequential(
            nn.Linear(out_dim, proj_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_out_dim, contrastive_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.dropout(h)
        h = torch.flatten(h, start_dim=1)
        return self.projector(h)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(z_i.size(0)).to(z_i.device)
        return nn.CrossEntropyLoss()(sim_matrix, labels)


def build_model(model_type: str, args, device: str):
    if model_type == 'GSDNN':
        encoder = GSDNN_new(
            args.num_classes, args.block_n, args.init_channels,
            args.growth_rate, args.base_channels, args.stride, args.dropout_GSDNN
        )
    elif model_type == 'ResNet':
        encoder = ResNet18()
    elif model_type == 'Conformer':
        encoder = Conformer(emb_size=40, depth=6, n_classes=4)
    elif model_type == 'Mamba':
        encoder = MambaV3(
            num_classes=args.num_classes,
            in_channels=args.init_channels,
            d_model=64,
            n_layers=4,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    model = SimCLRModel(encoder, args.out_dim, args.proj_out_dim,
                        args.contrastive_dim, args.dropout)
    return model.to(device)


# ==============================================================================
# Optimizer & Scheduler
# ==============================================================================

def get_warmup_lambda(warmup_steps: int):
    def lr_lambda(step: int):
        return step / warmup_steps if step < warmup_steps else 1.0
    return lr_lambda


def build_optimizer(model, args, total_steps: int):
    batch_size = args.batch_size
    base_lr = args.base_lr
    warmup_steps = int(total_steps * args.warmup_ratio)
    scaled_lr = base_lr * (batch_size / 256)
    min_lr = scaled_lr * 1e-3

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_lambda(warmup_steps))
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr,
    )
    return optimizer, warmup_scheduler, cosine_scheduler, warmup_steps


# ==============================================================================
# Training
# ==============================================================================

def train_epoch(model, dataloader, criterion, optimizer, warmup_scheduler,
                cosine_scheduler, warmup_steps, device, epoch, writer, args):
    model.train()
    total_loss = 0.0
    num_batches = 0
    steps_per_epoch = len(dataloader)

    for batch_idx, (img1, img2) in enumerate(dataloader):
        if args.model_type in ('GSDNN', 'Mamba'):
            img1 = img1.squeeze(1)
            img2 = img2.squeeze(1)

        img1 = img1.to(device, dtype=torch.float32)
        img2 = img2.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        loss = criterion(model(img1), model(img2))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step = epoch * steps_per_epoch + batch_idx + 1
        if step >= warmup_steps:
            cosine_scheduler.step()
        else:
            warmup_scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            global_step = epoch * steps_per_epoch + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        if args.mode == 'debug':
            break

    return total_loss / num_batches


# ==============================================================================
# Main
# ==============================================================================

def train(args):
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    print('Loading data...')
    dataloader = load_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        views=args.views,
        num_workers=args.num_workers,
        args=args
    )
    print(f'Data loaded: {len(dataloader.dataset)} samples, {len(dataloader)} batches')

    print(f'Building model: {args.model_type}')
    model = build_model(args.model_type, args, device)
    flops_input = [1, 18, 101] if args.model_type == 'Mamba' else [1, 1, 18, 101]
    FLOPs_calculat(model, device, flops_input)

    criterion = ContrastiveLoss(temperature=args.temperature).to(device)

    total_steps = args.epochs * len(dataloader)
    optimizer, warmup_scheduler, cosine_scheduler, warmup_steps = build_optimizer(
        model, args, total_steps
    )

    print('Starting training...')
    # loss_history = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            model, dataloader, criterion, optimizer, warmup_scheduler,
            cosine_scheduler, warmup_steps, device, epoch, writer, args
        )
        # loss_history.append(avg_loss)

        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        print(f'Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.6f}')

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_epoch = epoch
            save_checkpoint(model, args.save_dir, epoch, best_loss)

    print('Training completed!')
    print(f'Minimum loss: {best_loss:.6f} In epoch: {best_epoch}')

    writer.close()
    print(f'All results saved to: {args.save_dir}')
    print(f'View TensorBoard: tensorboard --logdir={args.log_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Gait self-supervised pre-training with SimCLR')
    parser.add_argument('--exp_name', type=str, default='Gait_selfsup_baseline')
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['normal', 'debug'], help='debug: run 1 batch per epoch')

    # Data
    parser.add_argument('--data_path', type=str, default='datasets/data_10000/all_data.mat')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0, help='0 in win, 4 in Linux')
    parser.add_argument('--views', type=int, default=2, help='Number of views for contrastive learning')

    # Model
    parser.add_argument('--model_type', type=str, default='ResNet',
                        choices=['GSDNN', 'ResNet', 'Conformer', 'Mamba'])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--block_n', type=int, default=8)
    parser.add_argument('--init_channels', type=int, default=18)
    parser.add_argument('--growth_rate', type=int, default=12)
    parser.add_argument('--base_channels', type=int, default=48)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--dropout_GSDNN', type=float, default=0.2)

    # Projection head
    parser.add_argument('--out_dim', type=int, default=512,
                        help='Encoder output dim (GSDNN=132, ResNet18=512)')
    parser.add_argument('--proj_out_dim', type=int, default=1024)
    parser.add_argument('--contrastive_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--base_lr', type=float, default=3e-4,
                        help='Base LR for batch size 256 (AdamW)')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW decoupled weight decay')

    # Augmentation
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6)

    # Save & Log
    parser.add_argument('--save_dir', type=str, default='./save_models')
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--save_freq', type=int, default=30)

    # Device
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('Gait Self-Supervised Pre-training Configuration')
    print('=' * 70)
    for key, value in sorted(vars(args).items()):
        print(f'  {key.ljust(30)}: {value}')
    print('=' * 70)

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
