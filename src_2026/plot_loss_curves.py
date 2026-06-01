"""
Loss curves pentru toate modelele fECG (Part 1 + Part 2).
Salvate in plots_2026/loss_plots/
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE   = '/shared_storage/iulia.orvas/paper/fECG_approx/models'
OUT    = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/loss_plots'
os.makedirs(OUT, exist_ok=True)

def load(path):
    return json.load(open(path))

def best_info(vals):
    idx = int(np.argmin(vals))
    return idx + 1, vals[idx]

# ============================================================
# PART 1 — Extraction models
# ============================================================

PART1 = [
    ('v1  (direct, SigMSE)',        f'{BASE}/movement_CUNet_128x400_composed_history.json',    'steelblue'),
    ('v5  (direct, AmpWeightedMSE)',f'{BASE}/movement_CUNet_128x400_ampw_history.json',         'teal'),
    ('v6  (direct, SigMSE+AmpW+BL)',f'{BASE}/movement_CUNet_128x400_v6_baseline_history.json',  'olive'),
    ('v7  (direct, SigMAE/L1)',     f'{BASE}/movement_CUNet_v7_paper_direct_history.json',       'saddlebrown'),
    ('v8  (direct, SigMSE, 1.87M)', f'{BASE}/movement_CUNet_v8_mse_history.json',               'tomato'),
    ('v9  (mask,   Sig+Cpl)',       f'{BASE}/movement_CUNet_v9_mask_history.json',               'darkorange'),
    ('v10 (mask,   Sig+Cpl+Peak)',  f'{BASE}/movement_CUNet_v10_fqrs_history.json',              'crimson'),
    ('v11 (direct, Sig+Peak)',      f'{BASE}/movement_CUNet_v11_direct_peak_history.json',       'mediumpurple'),
]

# --- Figure 1a: overview — 2×4 grid, total loss ---
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

for ax, (name, fpath, color) in zip(axes, PART1):
    h  = load(fpath)
    tr = h['train_loss']
    vl = h['val_loss']
    ep = list(range(1, len(tr) + 1))
    best_ep, best_val = best_info(vl)

    ax.plot(ep, tr, color=color, lw=1.2, alpha=0.75, label='train')
    ax.plot(ep, vl, color=color, lw=1.5, ls='--',    label='val')
    ax.axvline(best_ep, color='gray', lw=0.7, ls=':')
    ax.scatter([best_ep], [best_val], color=color, s=40, zorder=5)
    ax.set_title(f'{name}\nbest={best_val:.4f} @ ep{best_ep}', fontsize=8.5)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Loss',  fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)

fig.suptitle('Part 1 — fECG Extraction: toate modelele (total loss)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/part1_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: part1_all_models.png')

# --- Figure 1b: component breakdown v10 și v11 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

component_models = [
    ('v10 (mask, Sig+Cpl+Peak)', f'{BASE}/movement_CUNet_v10_fqrs_history.json',         'crimson',     ['sig','cpl','pk']),
    ('v11 (direct, Sig+Peak)',   f'{BASE}/movement_CUNet_v11_direct_peak_history.json',   'mediumpurple',['sig','pk']),
]
comp_colors = {'sig': 'steelblue', 'cpl': 'darkorange', 'pk': 'seagreen'}
comp_labels = {'sig': 'SignalMSE', 'cpl': 'ComplexMSE', 'pk': 'PeakMSE'}

for ax, (name, fpath, color, comps) in zip(axes, component_models):
    h  = load(fpath)
    ep = list(range(1, len(h['train_loss']) + 1))

    ax.plot(ep, h['train_loss'], color=color, lw=2.0,  label='total train')
    ax.plot(ep, h['val_loss'],   color=color, lw=2.0, ls='--', label='total val')
    for c in comps:
        cc = comp_colors[c]
        ax.plot(ep, h[f'train_{c}'], color=cc, lw=1.0, alpha=0.8, label=f'{comp_labels[c]} train')
        ax.plot(ep, h[f'val_{c}'],   color=cc, lw=1.0, alpha=0.8, ls='--', label=f'{comp_labels[c]} val')

    best_ep, best_val = best_info(h['val_loss'])
    ax.axvline(best_ep, color='gray', lw=0.7, ls=':')
    ax.set_title(f'{name}\nbest val={best_val:.4f} @ ep{best_ep}', fontsize=9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

fig.suptitle('Part 1 — Component loss breakdown: v10 și v11', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/part1_components_v10_v11.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: part1_components_v10_v11.png')

# --- Figure 1c: v1 vs v9 vs v11 — val loss suprapuse ---
fig, ax = plt.subplots(figsize=(10, 5))
overlay = [
    ('v1  (direct, SigMSE)',        f'{BASE}/movement_CUNet_128x400_composed_history.json',  'steelblue'),
    ('v9  (mask,   Sig+Cpl)',       f'{BASE}/movement_CUNet_v9_mask_history.json',            'darkorange'),
    ('v11 (direct, Sig+Peak)',      f'{BASE}/movement_CUNet_v11_direct_peak_history.json',    'mediumpurple'),
]
for name, fpath, color in overlay:
    h  = load(fpath)
    vl = h['val_loss']
    ep = list(range(1, len(vl) + 1))
    best_ep, best_val = best_info(vl)
    ax.plot(ep, vl, color=color, lw=1.8, label=f'{name}  best={best_val:.4f}@ep{best_ep}')
    ax.scatter([best_ep], [best_val], color=color, s=50, zorder=5)

ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Part 1 — Val loss: v1 vs v9 vs v11', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/part1_v1_v9_v11_val.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: part1_v1_v9_v11_val.png')

# ============================================================
# PART 2 — Classification models
# ============================================================

# modele cu train_loss/val_loss/train_acc/val_acc simple
CLF_SIMPLE = [
    ('clf_v9  (RF, 36feat)',       f'{BASE}/movement_clf_v9_history.json',   'steelblue'),
    ('clf_v10 (ResNet1D beats)',   f'{BASE}/movement_clf_v10_history.json',  'darkorange'),
    ('clf_v11 (Transformer beats)',f'{BASE}/movement_clf_v11_history.json',  'crimson'),
    ('clf_v12 (RF, 78feat)',       f'{BASE}/movement_clf_v12_history.json',  'teal'),
    ('clf_v13 (CNN+Transf QRS)',   f'{BASE}/movement_clf_v13_history.json',  'olive'),
    ('clf_v17 (SmallEnvResNet)',   f'{BASE}/movement_clf_v17_history.json',  'mediumpurple'),
]

# --- Figure 2a: grid loss + acc per model ---
n = len(CLF_SIMPLE)
fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n))

for row, (name, fpath, color) in enumerate(CLF_SIMPLE):
    h  = load(fpath)
    ep = list(range(1, len(h['train_loss']) + 1))

    ax_l = axes[row][0]
    ax_a = axes[row][1]

    # loss
    ax_l.plot(ep, h['train_loss'], color=color, lw=1.2, alpha=0.75, label='train')
    ax_l.plot(ep, h['val_loss'],   color=color, lw=1.5, ls='--',    label='val')
    best_ep_l, best_val_l = best_info(h['val_loss'])
    ax_l.axvline(best_ep_l, color='gray', lw=0.7, ls=':')
    ax_l.scatter([best_ep_l], [best_val_l], color=color, s=35, zorder=5)
    ax_l.set_ylabel(name, fontsize=8, color=color, fontweight='bold')
    ax_l.set_title(f'Loss  (best={best_val_l:.3f}@ep{best_ep_l})', fontsize=8)
    ax_l.legend(fontsize=7); ax_l.grid(True, alpha=0.25); ax_l.tick_params(labelsize=7)

    # accuracy
    tr_a = h['train_acc']; vl_a = h['val_acc']
    best_ep_a = int(np.argmax(vl_a)) + 1
    best_val_a = max(vl_a)
    ax_a.plot(ep, tr_a, color=color, lw=1.2, alpha=0.75, label='train')
    ax_a.plot(ep, vl_a, color=color, lw=1.5, ls='--',    label='val')
    ax_a.axvline(best_ep_a, color='gray', lw=0.7, ls=':')
    ax_a.scatter([best_ep_a], [best_val_a], color=color, s=35, zorder=5)
    ax_a.set_title(f'Accuracy  (best={best_val_a:.1f}%@ep{best_ep_a})', fontsize=8)
    ax_a.legend(fontsize=7); ax_a.grid(True, alpha=0.25); ax_a.tick_params(labelsize=7)

    if row == n - 1:
        ax_l.set_xlabel('Epoch', fontsize=8)
        ax_a.set_xlabel('Epoch', fontsize=8)

fig.suptitle('Part 2 — Classification: loss și accuracy per model', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/part2_clf_loss_acc.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: part2_clf_loss_acc.png')

# --- Figure 2b: val accuracy suprapuse — cele mai relevante modele ---
fig, ax = plt.subplots(figsize=(12, 5))
for name, fpath, color in CLF_SIMPLE:
    h  = load(fpath)
    vl_a = h['val_acc']
    ep   = list(range(1, len(vl_a) + 1))
    best_ep_a = int(np.argmax(vl_a)) + 1
    best_val_a = max(vl_a)
    ax.plot(ep, vl_a, color=color, lw=1.5, label=f'{name}  {best_val_a:.1f}%')
    ax.scatter([best_ep_a], [best_val_a], color=color, s=50, zorder=5)

ax.set_xlabel('Epoch'); ax.set_ylabel('Val Accuracy (%)')
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
ax.set_title('Part 2 — Val accuracy comparativ (neural models)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/part2_clf_val_acc_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: part2_clf_val_acc_overlay.png')

# --- Figure 2c: bar chart — best val accuracy toate modelele (neural + RF) ---
BAR_MODELS = [
    # (label, val_acc, is_rf)
    ('clf_v9\nRF 36feat',      92.20, True),
    ('clf_v10\nResNet1D',      90.90, False),
    ('clf_v11\nTransformer',   88.30, False),
    ('clf_v12\nRF 78feat',     91.24, True),
    ('clf_v13\nCNN+Transf',    76.20, False),
    ('clf_v14\nRF 71feat',     92.55, True),
    ('clf_v15\nRF+feat',       92.18, True),
    ('clf_v16\nEnvResNet',     88.93, False),
    ('clf_v17\nSmallEnvResNet',90.24, False),
    ('Ensemble\nRF+ResNet17',  93.07, False),
]

labels  = [m[0] for m in BAR_MODELS]
vals    = [m[1] for m in BAR_MODELS]
colors  = ['#2ecc71' if m[2] else '#3498db' for m in BAR_MODELS]
colors[-1] = 'gold'

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_ylim(70, 95)
ax.set_ylabel('Best Val Accuracy (%)', fontsize=10)
ax.set_title('Part 2 — Best val accuracy: toate modelele de clasificare\n'
             '(verde=RF, albastru=neural, auriu=ensemble)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', labelsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/part2_clf_best_acc_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: part2_clf_best_acc_bar.png')

# --- Figure 2d: clf_v16 și clf_v17 (EnvResNet) — componente separate ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

h16 = load(f'{BASE}/movement_clf_v16_history.json')
h17 = load(f'{BASE}/movement_clf_v17_history.json')

# v16 resnet
ep = list(range(1, len(h16['resnet_train_loss']) + 1))
ax = axes[0][0]
ax.plot(ep, h16['resnet_train_loss'], color='steelblue', lw=1.2, alpha=0.8, label='train')
ax.plot(ep, h16['resnet_val_loss'],   color='steelblue', lw=1.5, ls='--',   label='val')
best_ep, best_val = best_info(h16['resnet_val_loss'])
ax.axvline(best_ep, color='gray', lw=0.7, ls=':')
ax.set_title(f'clf_v16 EnvResNet — Loss\nbest={best_val:.3f}@ep{best_ep}', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

ax = axes[0][1]
best_ep_a = int(np.argmax(h16['resnet_val_acc'])) + 1
ax.plot(ep, h16['resnet_train_acc'], color='steelblue', lw=1.2, alpha=0.8, label='train')
ax.plot(ep, h16['resnet_val_acc'],   color='steelblue', lw=1.5, ls='--',   label='val')
ax.axvline(best_ep_a, color='gray', lw=0.7, ls=':')
ax.set_title(f'clf_v16 EnvResNet — Accuracy\nbest={max(h16["resnet_val_acc"]):.1f}%@ep{best_ep_a}', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# v17
ep = list(range(1, len(h17['train_loss']) + 1))
ax = axes[1][0]
ax.plot(ep, h17['train_loss'], color='mediumpurple', lw=1.2, alpha=0.8, label='train')
ax.plot(ep, h17['val_loss'],   color='mediumpurple', lw=1.5, ls='--',   label='val')
best_ep, best_val = best_info(h17['val_loss'])
ax.axvline(best_ep, color='gray', lw=0.7, ls=':')
ax.set_title(f'clf_v17 SmallEnvResNet — Loss\nbest={best_val:.3f}@ep{best_ep}', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

ax = axes[1][1]
ep_a = list(range(1, len(h17['val_acc']) + 1))
best_ep_a = int(np.argmax(h17['val_acc'])) + 1
ax.plot(ep_a, h17['train_acc'], color='mediumpurple', lw=1.2, alpha=0.8, label='train')
ax.plot(ep_a, h17['val_acc'],   color='mediumpurple', lw=1.5, ls='--',   label='val')
ax.axvline(best_ep_a, color='gray', lw=0.7, ls=':')
ax.set_title(f'clf_v17 SmallEnvResNet — Accuracy\nbest={max(h17["val_acc"]):.1f}%@ep{best_ep_a}', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

for ax in axes.flatten():
    ax.set_xlabel('Epoch', fontsize=8)

fig.suptitle('Part 2 — EnvResNet models: v16 vs v17', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/part2_envresnet_v16_v17.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: part2_envresnet_v16_v17.png')

print('\nDone. Toate ploturile salvate in:', OUT)
