"""Gait classification fine-tuning script."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.Gait_dataset_new import GaitDataModule, GaitDataset
from models.GSDNN_new import GSDNN_new
from models.resnet import ResNet18
from models.Conformer import Conformer
from models.mambav3 import MambaV3
from utils.analysis import set_seed
from utils.calculate import FLOPs_calculat
from utils.plt_curves import plot_confusion_matrix


# ==============================================================================
# Model
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, base_model: nn.Module, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return h


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_features: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def build_model(model_type: str, args, num_classes: int, device: str):
    if model_type == 'GSDNN':
        base = GSDNN_new(
            args.num_classes, args.block_n, args.init_channels,
            args.growth_rate, args.base_channels, args.stride, args.dropout_GSDNN
        )
    elif model_type == 'ResNet':
        base = ResNet18()
    elif model_type == 'Conformer':
        base = Conformer(emb_size=40, depth=6, n_classes=4)
    elif model_type == 'Mamba':
        base = MambaV3(
            num_classes=num_classes,
            in_channels=args.init_channels,
            d_model=64,
            n_layers=4,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    encoder = Encoder(base, dropout=args.dropout)
    model = Classifier(encoder, num_features=args.out_dim, num_classes=num_classes)
    return model.to(device)


# ==============================================================================
# Training & Evaluation
# ==============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, label in dataloader:
        if args.model_type in ('GSDNN', 'Mamba'):
            x = torch.squeeze(x, dim=1)
        x = x.to(device=device, dtype=torch.float32)
        label = label.to(device=device)

        optimizer.zero_grad()
        loss = criterion(model(x), label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if args.mode == 'debug':
            break

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device, model_type=''):
    model.eval()
    y_preds, y_labels, y_probs = [], [], []

    for x, y in dataloader:
        if model_type in ('GSDNN', 'Mamba'):
            x = torch.squeeze(x, dim=1)
        elif x.ndim != 4:
            x = torch.squeeze(x, dim=1)
        x = x.to(device=device, dtype=torch.float32)

        logits = model(x)
        y_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        y_labels.append(y.cpu().numpy())
        y_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    y_preds = np.concatenate(y_preds)
    y_labels = np.concatenate(y_labels)
    y_probs = np.concatenate(y_probs)

    acc = float(np.mean(y_preds == y_labels))
    precision = precision_score(y_labels, y_preds, average='macro', zero_division=0)
    recall = recall_score(y_labels, y_preds, average='macro', zero_division=0)
    f1 = f1_score(y_labels, y_preds, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_labels, y_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0

    return {
        'predictions': y_preds,
        'labels': y_labels,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


def log_metrics(writer, metrics, epoch, phase='train'):
    prefix = f'{phase}/'
    writer.add_scalar(f'{prefix}Accuracy', metrics['accuracy'], epoch)
    writer.add_scalar(f'{prefix}Precision', metrics['precision'], epoch)
    writer.add_scalar(f'{prefix}Recall', metrics['recall'], epoch)
    writer.add_scalar(f'{prefix}F1', metrics['f1'], epoch)
    writer.add_scalar(f'{prefix}AUC', metrics['auc'], epoch)
    if 'loss' in metrics:
        writer.add_scalar(f'{prefix}Loss', metrics['loss'], epoch)


def compute_final_metrics(history, last_n: int = 20):
    num_epochs = len(history['test_accs'])
    if num_epochs < last_n:
        print(f'\nWarning: Only {num_epochs} epochs trained, showing all epochs')
        last_n = num_epochs

    print(f"\n{'='*60}")
    print(f'Average metrics over last {last_n} epochs:')
    print(f"{'='*60}")
    print(f'Accuracy:  {np.mean(history["test_accs"][-last_n:]):.5f}')
    print(f'Precision: {np.mean(history["test_pres"][-last_n:]):.5f}')
    print(f'Recall:    {np.mean(history["test_recs"][-last_n:]):.5f}')
    print(f'F1:        {np.mean(history["test_f1s"][-last_n:]):.5f}')
    print(f'AUC:       {np.mean(history["test_aucs"][-last_n:]):.5f}')
    print(f"{'='*60}\n")


# ==============================================================================
# Main
# ==============================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Creating dataloaders...')
    train_loader, test_loader = GaitDataModule(args).setup()
    print(f'Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples')

    print(f'Building model: {args.model_type}')
    model = build_model(args.model_type, args, args.num_classes, device)

    if not args.supervised and args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f'Loading pretrained model from {args.pretrained_model}')
        kwargs = {'weights_only': True} if torch.__version__ >= '1.13.0' else {}
        state_dict = torch.load(args.pretrained_model, **kwargs)
        model.encoder.load_state_dict(state_dict['model_state_dict'], strict=False)

    if args.freeze_encoder:
        print('Freezing encoder parameters')
        for param in model.encoder.parameters():
            param.requires_grad = False
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    flops_input = [1, 18, 101] if args.model_type == 'Mamba' else [1, 1, 18, 101]
    FLOPs_calculat(model, device, flops_input)

    # Frozen-encoder fine-tuning: no weight decay; full fine-tuning / supervised: use weight_decay
    wd = 0.0 if args.freeze_encoder else args.weight_decay

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=wd,
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    print(f'TensorBoard logs: {args.log_dir}')
    print(f'Model save dir:   {args.save_dir}')

    history = {
        'losses': [],
        'train_accs': [], 'train_pres': [], 'train_recs': [], 'train_f1s': [], 'train_aucs': [],
        'test_accs': [], 'test_pres': [], 'test_recs': [], 'test_f1s': [], 'test_aucs': [],
    }
    best_acc = 0.0

    print(f'\nStarting training for {args.num_epochs} epochs...')
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args)

        train_metrics = evaluate(model, train_loader, device, args.model_type)
        test_metrics = evaluate(model, test_loader, device, args.model_type) if test_loader else None

        history['losses'].append(train_loss)
        history['train_accs'].append(train_metrics['accuracy'])
        history['train_pres'].append(train_metrics['precision'])
        history['train_recs'].append(train_metrics['recall'])
        history['train_f1s'].append(train_metrics['f1'])
        history['train_aucs'].append(train_metrics['auc'])

        if test_metrics is not None:
            history['test_accs'].append(test_metrics['accuracy'])
            history['test_pres'].append(test_metrics['precision'])
            history['test_recs'].append(test_metrics['recall'])
            history['test_f1s'].append(test_metrics['f1'])
            history['test_aucs'].append(test_metrics['auc'])
            log_metrics(writer, test_metrics, epoch, 'test')
        else:
            for key in ('test_accs', 'test_pres', 'test_recs', 'test_f1s', 'test_aucs'):
                history[key].append(0.0)

        log_metrics(writer, {**train_metrics, 'loss': train_loss}, epoch, 'train')

        if test_metrics is not None:
            print(f'Epoch [{epoch+1}/{args.num_epochs}] '
                  f'Loss: {train_loss:.5f} | '
                  f'Train Acc: {train_metrics["accuracy"]:.5f} | '
                  f'Test Acc: {test_metrics["accuracy"]:.5f}')
        else:
            print(f'Epoch [{epoch+1}/{args.num_epochs}] '
                  f'Loss: {train_loss:.5f} | '
                  f'Train Acc: {train_metrics["accuracy"]:.5f}')

        writer.add_scalar('train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

        current_acc = test_metrics['accuracy'] if test_metrics else train_metrics['accuracy']
        if current_acc > best_acc:
            best_acc = current_acc
            best_path = os.path.join(args.save_dir, f'best_model_{args.model_type}.pth')
            torch.save(model.state_dict(), best_path)
            print(f'  New best model saved with accuracy: {best_acc:.5f}')

    final_path = os.path.join(args.save_dir, f'final_acc_model_{args.model_type}.pth')
    torch.save(model.state_dict(), final_path)

    writer.close()

    if test_metrics is not None:
        compute_final_metrics(history)
        plot_confusion_matrix(
            test_metrics['labels'],
            test_metrics['predictions'],
            save_path=os.path.join(args.save_dir, 'confusion_matrix.png')
        )
    else:
        print(f"\n{'='*60}")
        print('Training completed (supervised mode, no test split)')
        print(f'Final train accuracy: {history["train_accs"][-1]:.5f}')
        print(f"{'='*60}\n")

    print(f'\nAll results saved to: {args.save_dir}')
    print(f'View TensorBoard: tensorboard --logdir={args.log_dir}')

    return history, model


def parse_args():
    parser = argparse.ArgumentParser(description='Gait classification fine-tuning')
    parser.add_argument('--exp_name', type=str, default='Gait_finetune')
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['normal', 'debug'], help='debug: run 1 batch per epoch')

    # Data
    parser.add_argument('--data_path', type=str, default='./datasets/data_10000/')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0, help='0 in win, 4 in Linux')

    # Training
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_classes', type=int, default=27)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW decoupled weight decay (ignored when freeze_encoder=True)')

    # Model
    parser.add_argument('--model_type', type=str, default='ResNet',
                        choices=['GSDNN', 'ResNet', 'Conformer', 'Mamba'])
    parser.add_argument('--pretrained_model', type=str,
                        default='')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--supervised', action='store_true',
                        help='Supervised learning mode (no pretraining, use all data for training)')

    # GSDNN-specific
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

    # Augmentation
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6)

    # Paths
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--save_dir', type=str, default='./save_models')

    args = parser.parse_args()

    print('=' * 70)
    print('Gait Fine-tuning Configuration')
    print('=' * 70)
    for key, value in sorted(vars(args).items()):
        print(f'  {key.ljust(30)}: {value}')
    print('=' * 70)

    return args


if __name__ == '__main__':
    args = parse_args()
    history, model = train(args)
    print('\nTraining completed!')
    print(f'Best test accuracy: {max(history["test_accs"]):.5f}')
