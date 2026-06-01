"""
Generate PDF report with model architecture and training tables (numbers only).
Output: plots_2026/model_report.pdf
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ARCH_DATA = {
    # cols: Model | Network | Params(M) | Conv type | Activation | Skip | Diagonal | Channels (enc→bottleneck)
    'headers': ['Model', 'Network', 'Params (M)', 'Conv type', 'Activation',
                'Skip conn.', 'Diagonal', 'Enc. channels'],
    'rows': [
        ['v1', 'ComplexUNet',   '0.59', 'Shared Re/Im', 'LeakyReLU(0.2)', 'Addition',     'Mag. scale exp(β)', '6→32→64→128'],
        ['v5', 'ComplexUNet',   '0.59', 'Shared Re/Im', 'LeakyReLU(0.2)', 'Addition',     'Mag. scale exp(β)', '6→32→64→128'],
        ['v6', 'ComplexUNet',   '0.59', 'Shared Re/Im', 'LeakyReLU(0.2)', 'Addition',     'Mag. scale exp(β)', '6→32→64→128'],
        ['v7', 'ComplexUNetV7', '1.87', 'Split Re/Im',  'RoActivation',   'Concatenation','Phase rot. e^{iβ}', '6→16→32→64→128'],
    ]
}

TRAIN_DATA = {
    'headers': ['Model', 'Loss function', 'Optimizer', 'LR', 'WD', 'BS',
                'Init', 'Max ep.', 'Patience', 'Epochs done',
                'Best val loss', 'Best ep.', 'Last val loss', 'Status'],
    'rows': [
        # v1
        ['v1', 'SignalMSE (L2)', 'AdamW', '1e-4', '1e-5', '32',
         'Random', '200', '15', '201',
         '0.014745', '199', '0.014968', 'Done'],
        # v5
        ['v5', 'SignalMSE + 3×AmpW', 'AdamW', '1e-4', '1e-5', '32',
         'Random', '200', '15', '154',
         '0.023846', '154', '0.023846', 'Running'],
        # v6
        ['v6', 'SignalMSE + AmpW + Baseline', 'AdamW', '1e-4', '1e-5', '32',
         'v1 warm', '200', '15', '42',
         '0.053158', '42', '0.053158', 'Running'],
        # v7
        ['v7', 'SignalMAE (L1)', 'Adam', '1e-4', '—', '16',
         'Random', '200', '15', '15',
         '0.085450', '15', '0.085450', 'Running'],
    ]
}

STFT_DATA = {
    'headers': ['n_fft', 'hop', 'win_len', 'FS (Hz)', 'Spec size (F×T)', 'ms/frame', 'Window (s)'],
    'rows': [
        ['256', '10', '128', '1000', '128×400', '10', '4'],
    ]
}

DATASET_DATA = {
    'headers': ['Split', 'Files', 'Segments', 'Notes'],
    'rows': [
        ['Train', '~83%', '~5220 batches (BS=32)', 'shuffled'],
        ['Val',   '~17%', '~922  batches (BS=32)', 'fixed'],
        ['Test',  '11',   'Sem1–Sem11 (Test_DB)',  'held-out, no GT fECG spec.'],
    ]
}

SNR_DATA = {
    'headers': ['Pair', 'SNR (dB)'],
    'rows': [
        ['fECG / mECG',      '-8.1'],
        ['fECG / mixture',   '-9.4'],
        ['fECG / movement',  '+4.0'],
        ['mECG / movement',  '+12.1'],
    ]
}

LOSS_DEF_DATA = {
    'headers': ['Name', 'Formula'],
    'rows': [
        ['SignalMSE',       'MSE(iSTFT(pred), y_time)'],
        ['SignalMAE',       'MAE(iSTFT(pred), y_time)'],
        ['AmpWeightedMSE',  'mean((pred−tgt)² · w),  w=(|tgt|/max)²'],
        ['BaselinePenalty', 'mean(pred² · (1−w)),     w=(|tgt|/max)²'],
    ]
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_table(ax, data, title, col_widths=None, fontsize=8.5, header_color='#2c5282', row_colors=None):
    ax.axis('off')
    headers = data['headers']
    rows    = data['rows']
    n_cols  = len(headers)
    n_rows  = len(rows)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    if row_colors is None:
        row_colors = [['#f7fafc' if i % 2 == 0 else '#ffffff'] * n_cols for i in range(n_rows)]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e0')
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(row_colors[row - 1][col])

    ax.set_title(title, fontsize=fontsize + 1.5, fontweight='bold', pad=8, loc='left')

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------

OUT_DIR = '../plots_2026'
os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'model_report.pdf')

with PdfPages(pdf_path) as pdf:

    # --- Page 1: Architecture + STFT + Loss defs ---
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.65,
                            top=0.93, bottom=0.04, left=0.02, right=0.98)

    ax_arch  = fig.add_subplot(gs[0])
    ax_stft  = fig.add_subplot(gs[1])
    ax_loss  = fig.add_subplot(gs[2])

    arch_widths = [0.055, 0.12, 0.085, 0.12, 0.12, 0.12, 0.165, 0.155]
    make_table(ax_arch, ARCH_DATA, 'Architecture', col_widths=arch_widths, fontsize=8)

    make_table(ax_stft, STFT_DATA, 'STFT Parameters', fontsize=9)

    make_table(ax_loss, LOSS_DEF_DATA, 'Loss Functions', fontsize=9,
               col_widths=[0.22, 0.78])

    fig.text(0.01, 0.97, 'fECG Extraction — Model Report', fontsize=13,
             fontweight='bold', va='top')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Page 2: Training table (wide) ---
    fig = plt.figure(figsize=(20, 7))
    ax  = fig.add_subplot(111)

    train_widths = [0.04, 0.16, 0.07, 0.045, 0.045, 0.04,
                    0.065, 0.055, 0.06, 0.07,
                    0.075, 0.055, 0.075, 0.06]
    make_table(ax, TRAIN_DATA, 'Training Configuration & Results', col_widths=train_widths, fontsize=8)

    # highlight "Running" rows
    table_objs = [c for c in ax.get_children() if hasattr(c, 'get_celld')]

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Page 3: Dataset + SNR ---
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55,
                            top=0.93, bottom=0.06, left=0.02, right=0.98)

    ax_ds  = fig.add_subplot(gs[0])
    ax_snr = fig.add_subplot(gs[1])

    make_table(ax_ds, DATASET_DATA, 'Dataset Split',
               col_widths=[0.1, 0.2, 0.35, 0.35], fontsize=9)
    make_table(ax_snr, SNR_DATA, 'Signal-to-Noise Ratios (training data)',
               col_widths=[0.5, 0.5], fontsize=9)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Page 4: Loss curves ---
    histories = {
        'v1 (SignalMSE)':              'models/movement_CUNet_128x400_composed_history.json',
        'v5 (SignalMSE+3×AmpW)':       'models/movement_CUNet_128x400_ampw_scratch_history.json',
        'v6 (SignalMSE+AmpW+Baseline)':'models/movement_CUNet_128x400_v6_baseline_history.json',
        'v7 (SignalMAE, paper arch)':   'models/movement_CUNet_v7_paper_direct_history.json',
    }
    colors = ['#2b6cb0', '#276749', '#c05621', '#6b46c1']

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(16, 5))
    for (label, path), color in zip(histories.items(), colors):
        full_path = os.path.join('..', path)
        if not os.path.exists(full_path):
            continue
        h = json.load(open(full_path))
        t = h.get('train_loss', [])
        v = h.get('val_loss',   [])
        if t: ax_train.plot(range(1, len(t)+1), t, label=label, color=color, lw=1.5)
        if v: ax_val.plot(  range(1, len(v)+1), v, label=label, color=color, lw=1.5)

    for ax, title in [(ax_train, 'Train Loss'), (ax_val, 'Val Loss')]:
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)

    fig.suptitle('Loss Curves', fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # metadata
    d = pdf.infodict()
    d['Title']  = 'fECG Model Report'
    d['Author'] = 'iulia.orvas'

print(f'Saved → {pdf_path}')
