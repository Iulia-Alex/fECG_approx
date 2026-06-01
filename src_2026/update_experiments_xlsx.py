"""
Rewrites experiments.xlsx with all up-to-date data (2026-04-26). All text in English.
"""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

PATH = '/shared_storage/iulia.orvas/paper/fECG_approx/experiments.xlsx'

# ── helpers ──────────────────────────────────────────────────────────────────
GRAY1  = PatternFill('solid', fgColor='D9D9D9')
GRAY2  = PatternFill('solid', fgColor='F2F2F2')
GREEN  = PatternFill('solid', fgColor='C6EFCE')
RED    = PatternFill('solid', fgColor='FFC7CE')
ORANGE = PatternFill('solid', fgColor='FFEB9C')
BLUE   = PatternFill('solid', fgColor='BDD7EE')

BOLD   = Font(bold=True)
WRAP   = Alignment(wrap_text=True, vertical='top')
CENTER = Alignment(horizontal='center', vertical='top', wrap_text=True)

thin   = Side(style='thin')
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

def header(ws, row, vals, fill=GRAY1):
    for c, v in enumerate(vals, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font      = BOLD
        cell.fill      = fill
        cell.alignment = CENTER
        cell.border    = BORDER

def row_data(ws, row, vals, fill=None, bold=False):
    for c, v in enumerate(vals, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.alignment = WRAP
        cell.border    = BORDER
        if fill: cell.fill = fill
        if bold: cell.font = BOLD

def title(ws, text, ncols=12):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    cell = ws.cell(row=1, column=1, value=text)
    cell.font      = Font(bold=True, size=13, color='FFFFFF')
    cell.fill      = PatternFill('solid', fgColor='1F4E79')
    cell.alignment = Alignment(horizontal='center', vertical='center')

def set_col_widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

# ── workbook ──────────────────────────────────────────────────────────────────
wb = openpyxl.Workbook()
wb.remove(wb.active)

# =============================================================================
# SHEET 1 — Model Overview
# =============================================================================
ws1 = wb.create_sheet('Model Overview')
title(ws1, 'fECG Extraction — Model Overview  |  updated 2026-05-18', ncols=12)

cols = ['Model', 'Status', 'Job', 'Node', 'Architecture', 'Params',
        'Output type', 'Loss Function', 'LR / BS', 'Init',
        'Best Val Loss', 'Inference Quality']
header(ws1, 2, cols)

N = '\n'  # explicit newline — wrap_text=True required on all cells

MODELS = [
    # (model, status, job, node, arch, params, output, loss, lr/bs, init, best_val, quality, fill)
    ('v1',
     f'DONE{N}ep 200/200{N}2026-03-23',
     '1276', 'Lenovo2',
     f'ComplexUNet (0.59M){N}Add skip, CReLU, Diag(mag)',
     '0.59 M', 'Direct',
     'SignalMSE',
     '1e-4 / 32', 'Random',
     '0.014745 @ ep 199',
     f'R-peaks ~90-95% GT{N}Clean baseline{N}Amplitude ceiling: sigmoid<=1 per bin',
     GREEN),
    ('v1 resume',
     f'DONE{N}ep 478/400{N}(continued past target)',
     '1357', 'Lenovo2',
     f'ComplexUNet (0.59M){N}continued from v1 ep200',
     '0.59 M', 'Direct',
     'SignalMSE',
     '1e-4 / 32', 'v1 best (ep199)',
     '0.013701 @ ep 458',
     f'Best signal quality among 1kHz models{N}Ran 478 total epochs{N}Superseded by v15 (0.0313 vs 0.0137 — different scale/pipeline)',
     GREEN),
    ('v2',
     f'DONE{N}ep 100/100',
     '1284', 'Lenovo6',
     f'Paper ComplexUNet (7.13M){N}Concat skip, RoActivation{N}Soft mask, 500 Hz',
     '7.13 M', 'Soft mask',
     'SignalMSE',
     '1e-4 / 32', 'Random',
     '0.03217 @ ep 99',
     f'R-peaks ~85-90% GT{N}More baseline noise than v1{N}12x params, no quality benefit',
     GRAY2),
    ('v3',
     f'CANCELLED{N}ep 101 — Amplitude suppression',
     '1287', 'Lenovo2',
     'ComplexUNet (=v1)',
     '0.59 M', 'Direct',
     f'SignalMSE{N}+ 5xPeakMSE(thresh){N}+ 0.1xComplexMSE',
     '1e-4 / 32', 'Random',
     '0.955 @ ep 103 (diff. scale)',
     f'R-peaks ~50-60% GT{N}ComplexMSE at 0.1x suppresses amplitudes',
     RED),
    ('v4',
     f'CANCELLED{N}ep 37 — Train/val gap',
     '1292', 'Lenovo2',
     'ComplexUNet (=v1)',
     '0.59 M', 'Direct',
     f'SignalMSE{N}+ 3xAmpWeightedMSE',
     '1e-5 / 32', 'Warm start v1',
     '0.02319 @ ep 36',
     f'Val plateau from ep1{N}Changing loss at fine-tuning = gradient landscape mismatch',
     RED),
    ('v5',
     f'DONE{N}ep 200/200',
     '1294', 'Lenovo2',
     'ComplexUNet (=v1)',
     '0.59 M', 'Direct',
     f'SignalMSE{N}+ 3xAmpWeightedMSE',
     '1e-4 / 32', 'Random',
     '0.022473 @ ep 200',
     f'R-peaks ~95% GT{N}Noisier baseline than v1{N}AmpW weights all large amplitudes (incl. artifacts)',
     GRAY2),
    ('v6',
     f'STOPPED{N}ep 67 — plateau',
     '1311', 'Lenovo2',
     'ComplexUNet (=v1)',
     '0.59 M', 'Direct',
     f'SignalMSE{N}+ AmpWeightedMSE{N}+ BaselinePenalty',
     '1e-4 / 32', 'Warm start v1',
     '0.052952 @ ep 67',
     f'Gradient landscape mismatch (warm start + diff. loss){N}Val flat 0.052-0.053, no improvement trend',
     RED),
    ('v7',
     f'STOPPED{N}ep 30 — L1 suppresses peaks',
     '1310', 'Lenovo6',
     f'ComplexUNetV7 (1.87M){N}Concat skip, RoActivation, Diag(phase)',
     '1.87 M', 'Direct',
     'SignalMAE (L1)',
     '1e-4 / 16', 'Random',
     '0.081819 @ ep 30',
     f'L1 suppresses R-peaks (sparse ~8% of signal){N}L1 minimises median error -> model ignores peaks',
     RED),
    ('v8',
     f'DONE{N}ep 56 (early stop){N}2026-04-07',
     '1319', 'Lenovo6',
     f'ComplexUNetV7 (1.87M){N}Concat skip, RoActivation, Diag(phase)',
     '1.87 M', 'Direct',
     'SignalMSE',
     '1e-4 / 16', 'Random',
     '0.046987 @ ep 41',
     f'Poor visually — direct pred without mask fails{N}Stronger arch (1.87M), worse than v1 (0.59M)',
     ORANGE),
    ('v9',
     f'DONE{N}ep 158 (early stop)',
     '1346', 'Lenovo6',
     f'ComplexUNetV9 (1.87M){N}Concat skip, RoActivation{N}Diag(phase), Soft mask',
     '1.87 M', 'Soft mask\nsigmoid x mixture',
     f'SignalMSE{N}+ ComplexMSE',
     '1e-4 / 16', 'Random',
     f'0.925510 @ ep 138{N}(early stop patience=20)',
     f'ComplexMSE dominates (~10x SignalMSE){N}Amplitudes suppressed{N}Same sigmoid<=1 ceiling',
     ORANGE),
    ('v10',
     f'DONE{N}ep 55 (early stop){N}2026-04-15',
     '1349', 'Lenovo2',
     f'ComplexUNetV10 (0.59M){N}Add skip, CReLU{N}Diag(mag), Soft mask',
     '0.59 M', 'Soft mask\nsigmoid x mixture',
     f'SignalMSE{N}+ ComplexMSE{N}+ 3xPeakMSE(fqrs)',
     '1e-4 / 32', 'Random',
     f'2.378023 @ ep 35{N}(cpl=1.626, sig=0.079, pk=0.233)',
     f'ComplexMSE=1.63 >> sig=0.08{N}PeakMSE cannot compensate{N}Amplitudes suppressed',
     RED),
    ('v11',
     f'STOPPED{N}ep 93/300{N}best @ ep73',
     '1359', 'Lenovo6',
     f'ComplexUNetV11 (1.87M){N}Concat skip, RoActivation{N}Diag(phase), Direct',
     '1.87 M', 'Direct (no mask)',
     f'SignalMSE{N}+ 3xPeakMSE(fqrs)',
     '1e-4 / 16', 'Random',
     f'0.432914 @ ep 73{N}(sig~0.047, pk~0.124)',
     f'No ComplexMSE -> no amplitude suppression{N}Direct pred harder to optimise than masked{N}Plateaued — superseded by v15',
     BLUE),
    ('v13',
     f'STOPPED{N}ep 84/300{N}best @ ep84{N}2026-05-18',
     '1378', 'Lenovo6',
     f'ComplexUNetV13 (1.87M){N}Concat skip, RoActivation{N}Diag(phase), InstanceNorm{N}1 kHz pipeline (128×400)',
     '1.87 M', 'Direct (no mask)',
     'SignalMSE',
     '1e-4 / 16', 'Random',
     '0.060014 @ ep 84',
     f'Plateau at val~0.060 — 2 root causes:{N}(1) InstNorm + mask on unnormalised input: scale-dependent{N}(2) Arch bottleneck 128ch vs v15 512ch{N}Cancelled to free GPU for v17',
     RED),
    ('v15',
     f'RUNNING{N}ep 95/300{N}best @ ep92{N}job 1381, Lenovo6',
     '1381', 'Lenovo6',
     f'ComplexUNet (7.13M){N}Concat skip, RoActivation{N}500 Hz native pipeline{N}128×128 STFT (paper-faithful)',
     '7.13 M', 'Direct (no mask)',
     'SignalMSE',
     '1e-4 / 32', 'Random',
     '0.031271 @ ep 92',
     f'Best signal quality so far{N}500Hz + 128×128 = exact paper spec{N}Converging slowly (~198 min/epoch)',
     GREEN),
    ('v16',
     f'RUNNING{N}ep 17/300{N}best @ ep15{N}job 1389, Lenovo2',
     '1389', 'Lenovo2',
     f'ComplexAttentionUNet (7.13M){N}= v15 + 2 Attention Gates (AG1, AG2){N}AG1: 64×64, AG2: 32×32{N}500 Hz, 128×128',
     '7.13 M', 'Direct (no mask)',
     'SignalMSE',
     '1e-4 / 32', 'Random',
     '0.032987 @ ep 15',
     f'Unsupervised attention gates{N}Val already < v13 final at ep15{N}Expected to match/beat v15 at convergence',
     GREEN),
    ('v17',
     f'RUNNING{N}ep 4/300{N}best @ ep2{N}job 1391, Lenovo6',
     '1391', 'Lenovo6',
     f'ComplexAttentionUNet (7.13M){N}= v16 + explicit attention supervision{N}Target mask from fqrs annotations{N}500 Hz, 128×128',
     '7.13 M', 'Direct (no mask)',
     f'SignalMSE{N}+ 0.1 × L_att{N}L_att = MSE(alpha_AG1, fqrs_mask)',
     '1e-4 / 32', 'Random',
     f'0.046913 total @ ep2{N}(sig_mse~0.049, l_att~0.046)',
     f'Attention gates guided to fQRS ±80ms windows{N}Hypothesis: faster/better QRS focus vs v16{N}Too early to evaluate — ep4 only',
     BLUE),
]

for i, m in enumerate(MODELS):
    r = i + 3
    row_data(ws1, r, m[:-1], fill=m[-1])

for r in range(2, len(MODELS) + 4):
    ws1.row_dimensions[r].height = 60

set_col_widths(ws1, [10, 18, 7, 9, 30, 8, 14, 28, 10, 14, 22, 48])


# =============================================================================
# SHEET 2 — Loss Functions
# =============================================================================
ws2 = wb.create_sheet('Loss Functions')
title(ws2, 'Loss Functions — Definitions, effects and lessons learned', ncols=5)

cols2 = ['Name', 'Formula', 'Domain', 'Effect on output', 'Used in']
header(ws2, 2, cols2, fill=GRAY1)

LOSSES = [
    ('SignalMSE',
     'mean( (iSTFT(pred_spec) - target_time)^2 )',
     'Time domain\n(after iSTFT)',
     'R-peak amplitudes ~90-95% GT\nSlightly noisy baseline\nNo amplitude suppression\n\nL2 penalises large errors quadratically -> forces correct peak heights',
     'v1, v2, v5, v8, v9, v11\n(baseline for all experiments)',
     GREEN),
    ('ComplexMSE',
     'mean((pred.real - tgt.real)^2)\n+ mean((pred.imag - tgt.imag)^2)',
     'Spectral\n(STFT domain)',
     'SUPPRESSES amplitudes even at 0.1x weight\n"Clean" baseline = near-zero output\nDo NOT combine with amplitude objectives\n\nAt convergence dominates 10-20x over SignalMSE\nregardless of architecture or other losses\n\nSymmetric w.r.t. phase -> model minimises by predicting\nlow-amplitude outputs that average out phase errors',
     'v3 (0.1x) CANCELLED\nv9 (1x) — dominates\nv10 (1x) — dominates equally',
     RED),
    ('PeakMSE (threshold)',
     'mask = |target| > 3*std, dilated +/-40 ms\nloss = MSE over masked positions',
     'Time domain\n(hard threshold)',
     'Arbitrary threshold misses small peaks\nCountered by ComplexMSE -> ineffective',
     'v3 (5x) CANCELLED',
     RED),
    ('AmpWeightedMSE',
     'w = (|target| / max|target|)^2\nloss = mean( (pred-target)^2 * w )',
     'Time domain\n(soft continuous weighting)',
     'w=0 on baseline, w=1 on R-peaks\nWeights ALL large-amplitude regions including mECG artifacts,\nnot only fetal R-peaks\nNoisier baseline than pure SignalMSE',
     'v4 (3x, warm start) CANCELLED\nv5 (3x, from scratch) DONE\nv6 (1x, warm start) STOPPED',
     ORANGE),
    ('PeakMSE (fqrs)',
     'mask = 1 at fqrs positions +/-30 samples, 0 elsewhere\nn = mask.sum() * n_channels\nloss = sum((pred-target)^2 * mask) / n',
     'Time domain\n(binary mask from .mat)',
     'Exact supervision at known fetal R-peak positions\nNo arbitrary threshold — fqrs from ground-truth .mat\nEffective ONLY without ComplexMSE\n\nv10: ComplexMSE=1.63 >> PeakMSE=0.23 -> ineffective\nv11: no ComplexMSE -> sig~0.047, pk~0.124 (still converging)',
     'v10 (3x) + ComplexMSE — INEFFECTIVE\nv11 (3x) without ComplexMSE — RUNNING',
     BLUE),
    ('SignalMAE (L1)',
     'mean( |iSTFT(pred_spec) - target_time| )',
     'Time domain',
     'Minimises median error -> ignores sparse peaks\nR-peaks ~8% of signal -> L1 drives model toward zero\nDo NOT use for signals with sparse high-amplitude peaks',
     'v7 STOPPED',
     RED),
]

for i, l in enumerate(LOSSES):
    r = i + 3
    row_data(ws2, r, l[:-1], fill=l[-1])

for r in range(2, len(LOSSES) + 4):
    ws2.row_dimensions[r].height = 85

set_col_widths(ws2, [18, 42, 14, 55, 30])


# =============================================================================
# SHEET 3 — Training Progress
# =============================================================================
ws3 = wb.create_sheet('Training Progress')
title(ws3, 'Training Progress — Val loss per epoch', ncols=11)

header(ws3, 2,
       ['Epoch', 'v1 val_loss', 'v1 resume val_loss', '',
        'Epoch', 'v9 val_loss', 'v11 val_loss (total / sig / pk)', '',
        'v13 val_loss', 'v15 val_loss', 'v16 val_loss'],
       fill=GRAY1)

V1_PROG = [
    (1,   0.08334,  None),
    (5,   0.04705,  None),
    (10,  0.03584,  None),
    (20,  0.02637,  None),
    (30,  0.02303,  None),
    (50,  0.02005,  None),
    (75,  0.01812,  None),
    (100, 0.01702,  None),
    (125, 0.01614,  None),
    (150, 0.01551,  None),
    (175, 0.01514,  None),
    (199, 0.014745, None),   # best v1
    (200, 0.014818, None),   # final v1
    (None, None, None),
    (210, None, 0.014701),
    (250, None, 0.014337),
    (300, None, 0.014159),
    (358, None, 0.013948),
    (365, None, 0.013937),
    (367, None, 0.013923),
    (372, None, 0.013908),
    (376, None, 0.013895),
    (379, None, 0.013873),
    (389, None, 0.013844),   # best resume
]

for i, (ep, v1, v1r) in enumerate(V1_PROG):
    r = i + 3
    fill = GREEN if ep in (199, 389) else None
    row_data(ws3, r, [ep, v1, v1r, None, None, None, None], fill=fill)

ws3.cell(row=14, column=1).value = '--- resume starts ---'
ws3.cell(row=14, column=1).font  = Font(italic=True, color='888888')
ws3.cell(row=14, column=2).value = 'v1 DONE ep199 best=0.014745'
ws3.cell(row=14, column=2).font  = Font(italic=True, color='888888')

# v9 progress (cols 5-6)
V9_PROG = [
    (1,   1.8970),
    (5,   1.0880),
    (10,  1.0107),
    (20,  0.9664),
    (50,  0.9429),
    (87,  0.9319),
    (95,  0.9318),
    (97,  0.9314),
    (110, 0.9271),
    (120, 0.9263),
    (130, 0.9258),
    (138, 0.9255),   # BEST
    (147, 0.9280),
    (155, None),     # running
]

for i, (ep, vl) in enumerate(V9_PROG):
    r = i + 3
    fill = GREEN if ep == 138 else None
    ws3.cell(row=r, column=5).value  = ep
    ws3.cell(row=r, column=6).value  = vl
    ws3.cell(row=r, column=5).border = BORDER
    ws3.cell(row=r, column=6).border = BORDER
    if fill:
        ws3.cell(row=r, column=5).fill = fill
        ws3.cell(row=r, column=6).fill = fill

# v11 progress (col 7) — total / sig / pk
V11_PROG = [
    (1,  '0.99 / 0.12 / 0.31'),
    (10, '0.72 / 0.088 / 0.21'),
    (20, '0.58 / 0.068 / 0.17'),
    (26, '0.458 / 0.056 / 0.134'),   # first local best
    (38, '0.472 / - / -'),            # early stop counter peaked at 12/20
    (40, '0.454 / - / -'),
    (42, '0.492 / - / -'),
    (43, '~0.49 / 0.049 / 0.131'),
    (49, '0.446 / ~0.047 / ~0.133'), # BEST
]

for i, (ep, vl) in enumerate(V11_PROG):
    r = i + 3
    fill = GREEN if ep == 49 else None
    ws3.cell(row=r, column=7).value     = f'ep{ep}: {vl}'
    ws3.cell(row=r, column=7).border    = BORDER
    ws3.cell(row=r, column=7).alignment = WRAP
    if fill:
        ws3.cell(row=r, column=7).fill = fill

# v13/v15/v16 progress (cols 9-11)
import json as _json
_base = '/shared_storage/iulia.orvas/paper/fECG_approx/models'
_h13 = _json.load(open(f'{_base}/movement_CUNet_v13_instanorm_history.json'))['val_loss']
_h15 = _json.load(open(f'{_base}/movement_CUNet_v15_paper_history.json'))['val_loss']
_h16 = _json.load(open(f'{_base}/movement_CUNet_v16_attention_history.json'))['val_loss']

_NEW = list(zip(
    _h13 + [None]*max(0, max(len(_h15), len(_h16)) - len(_h13)),
    _h15 + [None]*max(0, max(len(_h13), len(_h16)) - len(_h15)),
    _h16 + [None]*max(0, max(len(_h13), len(_h15)) - len(_h16)),
))

for i, (v13v, v15v, v16v) in enumerate(_NEW):
    r = i + 3
    best13 = min(_h13)
    best15 = min(_h15)
    best16 = min(_h16)
    for col, val, best in [(9, v13v, best13), (10, v15v, best15), (11, v16v, best16)]:
        cell = ws3.cell(row=r, column=col)
        cell.value     = round(val, 6) if val is not None else None
        cell.border    = BORDER
        cell.alignment = WRAP
        if val is not None and abs(val - best) < 1e-7:
            cell.fill = GREEN

for r in range(2, max(30, len(_NEW) + 4)):
    ws3.row_dimensions[r].height = 18

set_col_widths(ws3, [12, 16, 20, 4, 10, 16, 38, 4, 16, 16, 16])


# =============================================================================
# SHEET 4 — Architecture Details
# =============================================================================
ws4 = wb.create_sheet('Architecture Details')
title(ws4, 'Architecture Details — v1 through v11', ncols=8)

cols4 = ['Property', 'v1 / v5 (0.59M)', 'v8 (1.87M direct)', 'v9 (1.87M mask)',
         'v10 (0.59M mask)', 'v11 (1.87M direct)', 'v1 resume', 'Impact / Notes']
header(ws4, 2, cols4)

ARCH = [
    ('Params',
     '0.59 M', '1.87 M', '1.87 M', '0.59 M', '1.87 M', '0.59 M',
     '1.87M: deeper, concat skip, RoActivation\n0.59M: shallow, add skip, CReLU'),
    ('Output type',
     'Direct prediction', 'Direct prediction', 'Soft mask\nsigmoid x mixture', 'Soft mask\nsigmoid x mixture', 'Direct prediction', 'Direct prediction',
     'Mask = strong inductive prior, fast convergence\nDirect = no amplitude ceiling, but harder to optimise'),
    ('Amplitude ceiling',
     'sigmoid<=1\nmax output = max input (per bin)', 'N/A\n(direct pred)', 'sigmoid<=1\n(same)', 'sigmoid<=1\n(same)', 'None\n(direct pred)', 'sigmoid<=1\n(same)',
     'Mask cannot amplify beyond mixture per bin\nDirect prediction: can exceed mixture, but harder to train'),
    ('Complex conv',
     'Shared Re/Im\n(1 Conv2d)', 'Split Re/Im\n(conv_r + conv_i)', 'Split Re/Im', 'Shared Re/Im', 'Split Re/Im', 'Shared Re/Im',
     'Split more expressive, better phase capture'),
    ('Activation',
     'CReLU\n(ReLU on Re and Im)', 'RoActivation\n(CReLU+GK+GroupSort)', 'RoActivation', 'CReLU', 'RoActivation', 'CReLU',
     'RoActivation is phase-aware'),
    ('Skip connections',
     'Addition', 'Concatenation', 'Concatenation', 'Addition', 'Concatenation', 'Addition',
     'Concat preserves more information per layer'),
    ('Diagonal layer',
     'Diag(mag)\nexp(beta)*|x|', 'Diag(phase)\ne^{i*beta}*x', 'Diag(phase)', 'Diag(mag)', 'Diag(phase)', 'Diag(mag)',
     'Diag(mag): learnable magnitude scaling\nDiag(phase): learnable phase rotation'),
    ('Optimizer',
     'AdamW WD=1e-5', 'AdamW WD=1e-5', 'AdamW WD=1e-5', 'AdamW WD=1e-5', 'AdamW WD=1e-5', 'AdamW WD=1e-5',
     'Consistent across all models — grad clip=1.0'),
    ('Batch size',
     '32', '16', '16', '32', '16', '32',
     'BS=16 for 1.87M models (memory constraint)'),
    ('Best val loss',
     '0.013844\n(resume ep389)', '0.046987\nep41', '0.925510\nep138', '2.378023\nep35', '0.446454\nep49', '0.013844\nep389',
     'v1/resume best signal quality\nv9/v10 high val loss due to ComplexMSE scale — not directly comparable'),
    ('Status 2026-04-26',
     'DONE\n(ep200)', 'DONE\n(ep56)', 'RUNNING\nep155, job1346', 'DONE\n(ep55)', 'RUNNING\nep49, job1359', 'RUNNING\nep393, job1357',
     ''),
    ('Loss components\nat convergence',
     'sig=0.013844', 'sig=0.047', 'sig~0.09\ncpl~0.84', 'sig=0.079\ncpl=1.626\npk=0.233', 'sig~0.047\npk~0.124', 'sig=0.013844',
     'ComplexMSE scale explains high val loss in v9/v10\nNot directly comparable to pure SignalMSE models'),
]

FILLS4 = [None]*9 + [GREEN, BLUE, None]
for i, (row, fill) in enumerate(zip(ARCH, FILLS4)):
    r = i + 3
    row_data(ws4, r, list(row), fill=fill)

for r in range(2, len(ARCH) + 4):
    ws4.row_dimensions[r].height = 55

set_col_widths(ws4, [20, 22, 20, 20, 20, 20, 16, 48])


# =============================================================================
# SHEET 5 — Amplitude Analysis
# =============================================================================
ws5 = wb.create_sheet('Amplitude Analysis')
title(ws5, 'Amplitude Analysis — Why R-peak amplitudes plateau and how to fix it', ncols=4)

header(ws5, 2, ['Factor', 'Description', 'Affected models', 'Solution'], fill=GRAY1)

AMP = [
    ('Limit 1 — sigmoid <= 1\n(hard ceiling, architectural)',
     'Soft mask: fECG_pred = sigmoid(logits) x mixture_spec\nPer bin, output <= input AT ALL TIMES\n\nEven if mask = 1.0 at all R-peak bins,\noutput is bounded by the mixture amplitude at those frequencies.\nMixture contains overlapping mECG -> bins are "diluted".\n\nCANNOT be resolved by more training epochs or larger models.',
     'v1, v9, v10\n(all soft-mask models)',
     'Direct prediction (v11, v8)\nSoft mask with gain: alpha*sigmoid, alpha>1\nProposed v12: 1.5*sigmoid*mixture',
     RED),
    ('Limit 2 — Model capacity\n(soft ceiling, improvable)',
     'v1 (0.59M, add-skip, CReLU) vs v9 (1.87M, concat-skip, RoAct)\nSmaller model may miss complex spectral patterns\nThis limit is NOT independent of Limit 1\n\nTest: v9 with pure SignalMSE (no ComplexMSE) should outperform v1\nif capacity is a bottleneck. If it still plateaus at 90-95%, Limit 1 dominates.',
     'v1, v5 (0.59M)\nv10 (0.59M)',
     'v9-arch + pure SignalMSE (no ComplexMSE)\n= proposed v12 test',
     ORANGE),
    ('Limit 3 — ComplexMSE suppresses amplitudes\n(loss design issue)',
     'ComplexMSE at convergence dominates 10-20x over SignalMSE:\n  v9  ep138: cpl~0.84, sig~0.09  -> ratio ~9x\n  v10 ep35:  cpl=1.626, sig=0.079 -> ratio ~20x\n\nComplexMSE is symmetric w.r.t. phase -> model minimises\nby predicting low-amplitude outputs with average phase.\nSignalMSE directly penalises wrong peak heights in time domain.\n\nOriginal paper used SignalMSE only (confirmed from src/train.py)',
     'v9 (1x ComplexMSE)\nv10 (1x ComplexMSE + PeakMSE)',
     'Remove ComplexMSE entirely (v11)\nOr weight ComplexMSE << SignalMSE (0.01x?)',
     RED),
    ('Limit 4 — Direct prediction converges slower\n(optimisation difficulty)',
     'Soft mask: starts from mixture, learns what to keep\n  -> strong prior, fast convergence, good waveform shape\n\nDirect prediction: generates fECG from scratch\n  -> much larger search space, slower and less stable\n\nv11 ep49: val sig=0.047 vs v1 val=0.013 (3.5x worse)\nv8 ep56 (direct pred, 1.87M): val=0.047 -> poor visually',
     'v8, v11\n(direct prediction)',
     'More training epochs (v11 still converging)\nSoft mask with gain alpha>1 (best of both worlds)',
     ORANGE),
    ('Proposed solution — v12\nSoft mask with gain > 1',
     'mask = 1.5 * sigmoid(logits)    -> range [0, 1.5]\nfECG_pred = mask * mixture_spec\nloss = SignalMSE + 3*PeakMSE(fqrs)\n\nAdvantages over v11 (direct prediction):\n  - Retains "start from mixture" prior -> fast convergence like v1\n  - Mask can exceed 1.0 per bin -> can recover amplitudes above mixture\n  - PeakMSE explicitly supervises fQRS positions\n  - No ComplexMSE\n\nAdvantages over v9 (sigmoid mask):\n  - Removes sigmoid<=1 hard ceiling\n  - No ComplexMSE amplitude suppression',
     'Not yet implemented',
     'Launch after v11 finishes',
     BLUE),
]

for i, row in enumerate(AMP):
    r = i + 3
    row_data(ws5, r, row[:-1], fill=row[-1])

for r in range(2, len(AMP) + 4):
    ws5.row_dimensions[r].height = 105

set_col_widths(ws5, [30, 62, 22, 40])


# =============================================================================
# SHEET 6 — Part 2 Classification
# =============================================================================
ws6 = wb.create_sheet('Part2 Classification')
title(ws6, 'Part 2 — Movement Classification  |  Best: Ensemble RF+ResNet17 = 93.07%', ncols=6)

cols6 = ['Model', 'Input', 'Architecture', 'Val Acc', 'Notes', 'Status']
header(ws6, 2, cols6)

NL = '\n'   # explicit newline — cells must have wrap_text=True

CLF = [
    ('clf_v9',
     '36 spectral features per complete phase',
     'Random Forest', '92.20%',
     f'fft_freq = dominant feature (top 5-7){NL}Frequency of beat amplitude oscillation encodes movement type',
     'DONE', GREEN),
    ('clf_v10',
     'Beat amplitudes per phase (6, 200) padded',
     'ResNet1D', '90.90%',
     'Overfit: train 99% vs val 91%', 'DONE', ORANGE),
    ('clf_v11',
     'Beat amplitudes per phase',
     'Transformer', '88.30%',
     'No overfit — val accuracy plateaus', 'DONE', GRAY2),
    ('clf_v12',
     '78 features (36 + RR + QRS PCA + envelope)',
     'Random Forest', '91.24%',
     f'New features = noise, no new information{NL}0 new features in top-15 RF importance', 'DONE', GRAY2),
    ('clf_v13',
     'QRS morphology (150, 6, 50)',
     'CNN+Transformer', '76.20%',
     'Isolated beat shape is not discriminative for movement type', 'DONE', RED),
    ('clf_v14',
     '71 features (36 + cross-channel cross-correlations)',
     'Random Forest', '92.55%',
     f'Best RF standalone{NL}Cross-correlations provide marginal gain over v9', 'DONE', GREEN),
    ('clf_v15',
     '72 features (36 + phase_diff + coherence)',
     'Random Forest', '92.18%',
     'Weaker than v14 — inter-channel coherence adds no info', 'DONE', GRAY2),
    ('clf_v16',
     'Hilbert envelope (6, 200)',
     'EnvResNet (500K)', '88.93%',
     'Severe overfit: train 99.6% vs val 89%', 'DONE', ORANGE),
    ('clf_v17',
     'Hilbert envelope (6, 200)',
     'SmallEnvResNet (60K)', '90.24%',
     f'No overfit (aggressive augmentation){NL}train 91.8% ~ val 90.24%', 'DONE', GREEN),
    ('ENSEMBLE  RF(w=0.75) + ResNet17(w=0.25)',
     '36 spectral features + Hilbert envelope',
     'Weighted soft vote', '93.07%',
     f'BEST OVERALL{NL}Linear <-> Helical = only remaining confusion (~120/170 errors){NL}Stationary and Screw nearly perfect',
     'DONE', GREEN),
]

for i, c in enumerate(CLF):
    r = i + 3
    fill = c[-1]
    row_data(ws6, r, c[:-1], fill=fill)
    if 'ENSEMBLE' in c[0]:
        for col in range(1, 7):
            ws6.cell(row=r, column=col).font = Font(bold=True)

# Confusion matrix — row 13, no merge (avoids border artefacts)
CM_TEXT = (
    'Confusion Matrix — Ensemble val (2026-04-08):'
    '   Stationary: 751/752   |   Screw: 401/401'
    '   |   Linear: 356/401   |   Helical: 266/352'
)
for col in range(1, 7):
    cell = ws6.cell(row=13, column=col)
    cell.fill      = BLUE
    cell.border    = Border(left=Side(style='thin'), right=Side(style='thin'),
                            top=Side(style='thin'),  bottom=Side(style='thin'))
    cell.alignment = Alignment(wrap_text=True, vertical='top')
ws6.cell(row=13, column=1).value = CM_TEXT
ws6.cell(row=13, column=1).font  = Font(bold=True)
ws6.merge_cells(start_row=13, start_column=1, end_row=13, end_column=6)
ws6.row_dimensions[13].height = 24

for r in range(2, 13):
    ws6.row_dimensions[r].height = 52

set_col_widths(ws6, [30, 30, 20, 10, 52, 8])

# ── save ──────────────────────────────────────────────────────────────────────
wb.save(PATH)
print(f'Saved: {PATH}')
print(f'Sheets: {wb.sheetnames}')
