"""
Adds two new sheets to experiments.xlsx:
  - "Loss Curves"      : all loss/accuracy plots (Part 1 + Part 2)
  - "Inference Plots"  : Test DB inference grids (Sem1/2/3, all models)
"""
import openpyxl
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage

XLSX  = '/shared_storage/iulia.orvas/paper/fECG_approx/experiments.xlsx'
LDIR  = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/loss_plots'
IDIR  = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb'

TARGET_W = 820   # target pixel width for all images in Excel
ROW_H_PX = 15.0  # approx pixels per row (Excel default ~15pt ≈ 20px, but empirically 15 works)

def scaled_dims(path, target_w=TARGET_W):
    """Return (width_px, height_px) scaled to target_w."""
    img = PILImage.open(path)
    w, h = img.size
    scale = target_w / w
    return int(target_w), int(h * scale)

def rows_for(h_px, row_h=ROW_H_PX):
    return max(1, int(h_px / row_h) + 2)

def add_image(ws, path, anchor_row, target_w=TARGET_W):
    """Add image at given row, col A. Returns number of rows consumed."""
    w, h = scaled_dims(path, target_w)
    img = XLImage(path)
    img.width  = w
    img.height = h
    img.anchor = f'A{anchor_row}'
    ws.add_image(img)
    return rows_for(h)

def section_label(ws, row, text, subtitle=None):
    """Write a bold section header."""
    cell = ws.cell(row=row, column=1, value=text)
    cell.font      = Font(bold=True, size=11)
    cell.fill      = PatternFill('solid', fgColor='1F4E79')
    cell.font      = Font(bold=True, size=11, color='FFFFFF')
    cell.alignment = Alignment(wrap_text=False)
    ws.row_dimensions[row].height = 20
    if subtitle:
        row += 1
        c2 = ws.cell(row=row, column=1, value=subtitle)
        c2.font = Font(italic=True, size=9, color='444444')
        ws.row_dimensions[row].height = 14
        return row + 1
    return row + 1

def sheet_title(ws, text, ncols=8):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    cell = ws.cell(row=1, column=1, value=text)
    cell.font      = Font(bold=True, size=13, color='FFFFFF')
    cell.fill      = PatternFill('solid', fgColor='1F4E79')
    cell.alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 24
    ws.column_dimensions['A'].width = 120  # wide enough for images

# ── load workbook ─────────────────────────────────────────────────────────────
wb = openpyxl.load_workbook(XLSX)

# remove old sheet versions if re-running
for name in ['Loss Curves', 'Inference Plots',
             'Loss Curves (Part 1)', 'Loss Curves (Part 2)',
             'Inference (v1-v11)', 'Inference (v15-v17)']:
    if name in wb.sheetnames:
        del wb[name]

# =============================================================================
# SHEET — Loss Curves (Part 1)  — extraction models
# =============================================================================
ws_l1 = wb.create_sheet('Loss Curves (Part 1)')
sheet_title(ws_l1, 'Loss Curves — Part 1: fECG Extraction  |  v1 through v17')
cur = 2

cur = section_label(ws_l1, cur,
    'All extraction models (v1-v11): training and validation loss',
    subtitle='v1 (direct, SigMSE) | v5 (AmpW) | v7 (L1) | v8 (direct 1.87M) | '
             'v9 (mask Sig+Cpl) | v10 (mask Sig+Cpl+Peak) | v11 (direct Sig+Peak)')
cur += add_image(ws_l1, f'{LDIR}/part1_all_models.png', cur)
cur += 1

cur = section_label(ws_l1, cur,
    'Loss component breakdown: v10 (ComplexMSE dominates) and v11 (no ComplexMSE)',
    subtitle='v10: cpl=1.626 >> sig=0.079  |  v11: sig~0.047, pk~0.124 — no ComplexMSE')
cur += add_image(ws_l1, f'{LDIR}/part1_components_v10_v11.png', cur)
cur += 1

cur = section_label(ws_l1, cur,
    'Validation loss overlay: v1-resume (DONE ep458) vs v9 (DONE ep158) vs v11 (DONE ep93)',
    subtitle='v1-resume best=0.01370 @ ep458  |  v9 best=0.9255 (includes ComplexMSE — diff. scale)  |  v11 best=0.4329 @ ep73')
cur += add_image(ws_l1, f'{LDIR}/part1_v1_v9_v11_val.png', cur)
cur += 1

cur = section_label(ws_l1, cur,
    'New models: v13 / v15 / v16 / v17 — validation loss overlay + v17 attention components',
    subtitle='v13 STOPPED ep84 best=0.0600 | v15 RUNNING ep95 best=0.0313 | '
             'v16 RUNNING ep17 best=0.0330 | v17 RUNNING ep4 total=0.0469')
cur += add_image(ws_l1, f'{LDIR}/part1_v13_v15_v16_v17.png', cur)
cur += 1

cur = section_label(ws_l1, cur,
    'Train vs Val per model: v13 / v15 / v16 / v17',
    subtitle='v13 STOPPED ep84 | v15 RUNNING ep95 | v16 RUNNING ep17 | v17 RUNNING ep4')
cur += add_image(ws_l1, f'{LDIR}/part1_train_val_v13_v15_v16_v17.png', cur)
print(f'Loss Curves (Part 1): {cur} rows used')

# =============================================================================
# SHEET — Loss Curves (Part 2)  — classification models
# =============================================================================
ws_l2 = wb.create_sheet('Loss Curves (Part 2)')
sheet_title(ws_l2, 'Loss Curves — Part 2: Movement Classification  |  Best: Ensemble 93.07%')
cur = 2

cur = section_label(ws_l2, cur,
    'Best validation accuracy per model (all classifiers)',
    subtitle='Green = RF  |  Blue = neural  |  Gold = ensemble  |  Best: RF(0.75)+ResNet17(0.25) = 93.07%')
cur += add_image(ws_l2, f'{LDIR}/part2_clf_best_acc_bar.png', cur)
cur += 1

cur = section_label(ws_l2, cur,
    'Validation accuracy over epochs: neural models',
    subtitle='clf_v10 ResNet1D 90.9% | clf_v11 Transformer 88.3% | clf_v13 CNN+Transf 76.2% | clf_v17 SmallEnvResNet 90.2%')
cur += add_image(ws_l2, f'{LDIR}/part2_clf_val_acc_overlay.png', cur)
cur += 1

cur = section_label(ws_l2, cur,
    'Loss and accuracy curves per model (neural classifiers)',
    subtitle='Left: val loss  |  Right: val accuracy  |  Models: clf_v9/v10/v11/v12/v13/v17')
cur += add_image(ws_l2, f'{LDIR}/part2_clf_loss_acc.png', cur)
cur += 1

cur = section_label(ws_l2, cur,
    'EnvResNet: clf_v16 (500K, overfit) vs clf_v17 SmallEnvResNet (60K, no overfit)',
    subtitle='v16: train 99.6% vs val 89%  |  v17: train 91.8% ~ val 90.2%')
cur += add_image(ws_l2, f'{LDIR}/part2_envresnet_v16_v17.png', cur)
print(f'Loss Curves (Part 2): {cur} rows used')

# =============================================================================
# SHEET — Inference (v1-v11)
# =============================================================================
ws_i1 = wb.create_sheet('Inference (v1-v11)')
sheet_title(ws_i1,
    'Inference on Test DB — v1 / v9 / v10 / v11 vs Ground Truth  |  Window 10-14s @ 1kHz')
ws_i1.column_dimensions['A'].width = 120
cur = 2

for stem, snr in [('Sem1','7.8 dB'), ('Sem2','-9.7 dB'), ('Sem3','4.7 dB')]:
    cur = section_label(ws_i1, cur,
        f'{stem} (SNR≈{snr}) — v1-resume / v9 / v10 / v11 vs GT',
        subtitle='Rows: GT (green) | v1-resume ep458 best=0.01370 | v9 ep138 best=0.9255 | '
                 'v10 ep35 best=2.378 | v11 ep73 best=0.4329')
    cur += add_image(ws_i1, f'{IDIR}/{stem}_all_models_grid.png', cur)
    cur += 2
print(f'Inference (v1-v11): {cur} rows used')

# =============================================================================
# SHEET — Inference (v15-v17)
# =============================================================================
ws_i2 = wb.create_sheet('Inference (v15-v17)')
sheet_title(ws_i2,
    'Inference on Test DB — v15 / v16 / v17 vs Ground Truth  |  Window 10-13.83s @ 500Hz')
ws_i2.column_dimensions['A'].width = 120
cur = 2

for stem, snr, note in [
    ('Sem1', '7.8 dB',  'within training SNR range (0-6 dB)'),
    ('Sem2', '-9.7 dB', 'OUT of training range — all models struggle'),
    ('Sem3', '4.7 dB',  'best SNR match to training distribution'),
]:
    cur = section_label(ws_i2, cur,
        f'{stem} (SNR≈{snr}) — v15 / v16 / v17 vs GT  [{note}]',
        subtitle='Rows: GT (green) | v15 ep95 val=0.0313 | v16 ep17 val=0.0330 | v17 ep4 total=0.0469')
    cur += add_image(ws_i2, f'{IDIR}/{stem}_v15_v16_v17.png', cur)
    cur += 2
print(f'Inference (v15-v17): {cur} rows used')

# ── save ──────────────────────────────────────────────────────────────────────
wb.save(XLSX)
print(f'\nSaved: {XLSX}')
print(f'All sheets: {wb.sheetnames}')
