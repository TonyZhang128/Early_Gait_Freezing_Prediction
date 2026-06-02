"""
Gait Signal Visualization Tool
==============================
Visualize gait signals from .mat data files and save plots to local directory.

Usage:
    python gait_plot.py --data_path datasets/data_10000 --output_dir plots
    python gait_plot.py --mode single --subject_id 1 --channels 0,1,2
    python gait_plot.py --mode compare --subjects 1,2,3
"""

import argparse
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def load_data(data_path: str):
    """Load gait data and labels from .mat files."""
    sub_data = sio.loadmat(f'{data_path}/sub_data.mat')['sub_data']
    sub_label = sio.loadmat(f'{data_path}/sub_label.mat')['sub_label'][0]
    all_data = sio.loadmat(f'{data_path}/all_data.mat')['all_data']
    return sub_data, sub_label, all_data


def get_subject_data(data: np.ndarray, labels: np.ndarray, subject_id: int):
    """Extract data for a specific subject (1-based ID)."""
    mask = labels == subject_id
    return data[mask]


def plot_single_subject(data: np.ndarray, subject_id: int, channels: list[int],
                        output_dir: str, dpi: int = 150):
    """Plot selected channels for a single subject."""
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)

    if n_channels == 1:
        axes = [axes]

    for i, ch in enumerate(channels):
        axes[i].plot(data[0, ch, :], linewidth=1.5, color='steelblue')
        axes[i].set_ylabel(f'Channel {ch}')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    fig.suptitle(f'Subject {subject_id} - Gait Signal', fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'subject_{subject_id}_channels.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_all_channels(data: np.ndarray, subject_id: int,
                      output_dir: str, dpi: int = 150):
    """Plot all 18 channels vertically stacked in blue."""
    fig, axes = plt.subplots(18, 1, figsize=(14, 14), sharex=True)

    for ch in range(18):
        axes[ch].plot(data[0, ch, :], linewidth=1.4, color='#2980b9', alpha=0.85)
        axes[ch].set_ylabel(ch, fontsize=9, rotation=0, labelpad=20, fontweight='bold')
        axes[ch].grid(False)
        axes[ch].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
        for spine in axes[ch].spines.values():
            spine.set_visible(False)
        axes[ch].set_facecolor('none')

    axes[-1].tick_params(axis='x', which='both', length=0, labelbottom=False)
    plt.subplots_adjust(hspace=0.12, left=0.06, right=0.97, top=0.99, bottom=0.02)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'subject_{subject_id}_all_channels.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {save_path}')


def plot_subject_comparison(data: np.ndarray, labels: np.ndarray,
                            subject_ids: list[int], channel: int,
                            output_dir: str, dpi: int = 150):
    """Compare same channel across multiple subjects."""
    n_subjects = len(subject_ids)
    fig, axes = plt.subplots(n_subjects, 1, figsize=(12, 3 * n_subjects), sharex=True)

    if n_subjects == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_subjects))

    for i, sid in enumerate(subject_ids):
        subject_data = get_subject_data(data, labels, sid)
        if len(subject_data) == 0:
            print(f'Warning: No data for subject {sid}')
            continue
        axes[i].plot(subject_data[0, channel, :], linewidth=1.5, color=colors[i])
        axes[i].set_ylabel(f'Subject {sid}')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    fig.suptitle(f'Channel {channel} Comparison Across Subjects', fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'channel_{channel}_comparison.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_mean_signals(data: np.ndarray, labels: np.ndarray,
                      subject_ids: list[int], channel: int,
                      output_dir: str, dpi: int = 150):
    """Plot mean signal per subject on the same axes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(subject_ids)))

    for i, sid in enumerate(subject_ids):
        subject_data = get_subject_data(data, labels, sid)
        if len(subject_data) == 0:
            continue
        mean_signal = subject_data[:, channel, :].mean(axis=0)
        ax.plot(mean_signal, linewidth=1.5, color=colors[i], label=f'Subject {sid}')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Signal Value')
    ax.set_title(f'Channel {channel} - Mean Signal Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'channel_{channel}_mean_comparison.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_heatmap(data: np.ndarray, subject_id: int,
                 output_dir: str, dpi: int = 150):
    """Plot signal heatmap (channels x time)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data[0], aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Channel')
    ax.set_title(f'Subject {subject_id} - Signal Heatmap')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'subject_{subject_id}_heatmap.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_initial_signals(data: np.ndarray, subject_id: int,
                        output_dir: str, dpi: int = 150):
    """Plot 6 initial signals by summing high+mid+low freq channels."""
    # 6 initial channels: each = sum of 3 freq bands (high+mid+low)
    # ch0 = ch0+ch1+ch2, ch1 = ch3+ch4+ch5, ..., ch5 = ch15+ch16+ch17
    n_initial = 6
    initial_signals = np.zeros((n_initial, data.shape[2]))

    for i in range(n_initial):
        high_ch = i * 3      # high freq
        mid_ch = i * 3 + 1   # mid freq
        low_ch = i * 3 + 2   # low freq
        initial_signals[i] = data[0, high_ch] + data[0, mid_ch] + data[0, low_ch]

    fig, axes = plt.subplots(n_initial, 1, figsize=(16, 8), sharex=True)

    for i in range(n_initial):
        axes[i].plot(initial_signals[i], linewidth=1.0, color='steelblue')
        axes[i].set_ylabel(i, fontsize=9, rotation=0, labelpad=20, fontweight='bold')
        axes[i].grid(False)
        axes[i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
        for spine in axes[i].spines.values():
            spine.set_visible(False)
        axes[i].set_facecolor('none')

    axes[-1].tick_params(axis='x', which='both', length=0, labelbottom=False)
    plt.subplots_adjust(hspace=0.2, left=0.05, right=0.97, top=0.99, bottom=0.02)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'subject_{subject_id}_initial_signals.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {save_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Gait Signal Visualization')
    parser.add_argument('--data_path', type=str, default='datasets/data_10000',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Output directory for plots')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'all_channels', 'compare', 'mean', 'heatmap', 'initial', 'all'],
                        help='Visualization mode')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='Subject ID for single subject plots')
    parser.add_argument('--subjects', type=str, default='1,2,3',
                        help='Comma-separated subject IDs for comparison')
    parser.add_argument('--channels', type=str, default='0,1,2',
                        help='Comma-separated channel indices')
    parser.add_argument('--channel', type=int, default=0,
                        help='Single channel index for comparison')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output image DPI')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    sub_data, sub_label, all_data = load_data(args.data_path)
    print(f'Loaded: sub_data={sub_data.shape}, labels={sub_label.shape}, all_data={all_data.shape}')

    # Parse lists
    channels = [int(c) for c in args.channels.split(',')]
    subjects = [int(s) for s in args.subjects.split(',')]

    # Generate plots based on mode
    if args.mode == 'single':
        subject_data = get_subject_data(sub_data, sub_label, args.subject_id)
        if len(subject_data) == 0:
            print(f'Error: No data for subject {args.subject_id}')
            return
        plot_single_subject(subject_data, args.subject_id, channels, args.output_dir, args.dpi)

    elif args.mode == 'all_channels':
        subject_data = get_subject_data(sub_data, sub_label, args.subject_id)
        if len(subject_data) == 0:
            print(f'Error: No data for subject {args.subject_id}')
            return
        plot_all_channels(subject_data, args.subject_id, args.output_dir, args.dpi)

    elif args.mode == 'compare':
        plot_subject_comparison(sub_data, sub_label, subjects, args.channel, args.output_dir, args.dpi)

    elif args.mode == 'mean':
        plot_mean_signals(sub_data, sub_label, subjects, args.channel, args.output_dir, args.dpi)

    elif args.mode == 'heatmap':
        subject_data = get_subject_data(sub_data, sub_label, args.subject_id)
        if len(subject_data) == 0:
            print(f'Error: No data for subject {args.subject_id}')
            return
        plot_heatmap(subject_data, args.subject_id, args.output_dir, args.dpi)

    elif args.mode == 'initial':
        subject_data = get_subject_data(sub_data, sub_label, args.subject_id)
        if len(subject_data) == 0:
            print(f'Error: No data for subject {args.subject_id}')
            return
        plot_initial_signals(subject_data, args.subject_id, args.output_dir, args.dpi)

    elif args.mode == 'all':
        # Generate all plot types for subject 1
        subject_data = get_subject_data(sub_data, sub_label, args.subject_id)
        if len(subject_data) == 0:
            print(f'Error: No data for subject {args.subject_id}')
            return
        plot_single_subject(subject_data, args.subject_id, channels, args.output_dir, args.dpi)
        plot_all_channels(subject_data, args.subject_id, args.output_dir, args.dpi)
        plot_initial_signals(subject_data, args.subject_id, args.output_dir, args.dpi)
        plot_heatmap(subject_data, args.subject_id, args.output_dir, args.dpi)
        plot_subject_comparison(sub_data, sub_label, subjects, args.channel, args.output_dir, args.dpi)
        plot_mean_signals(sub_data, sub_label, subjects, args.channel, args.output_dir, args.dpi)

    print('Done.')


if __name__ == '__main__':
    main()
