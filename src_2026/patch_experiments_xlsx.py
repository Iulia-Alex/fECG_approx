"""
Patch experiments.xlsx — 2026-04-27.
Updates only cells that changed since 2026-04-26.
Does NOT touch image sheets ('Loss Curves', 'Inference Plots').

Changes:
  Sheet 1 (Model Overview):
    - v1 resume: DONE ep399, best=0.013776
    - v9:        DONE ep138
    - v11:       RUNNING ep59+, early stop 9/20
    - v12:       new row (job 1371, Lenovo2)
  Sheet 2 (Loss Functions):
    - PeakMSE(fqrs): update 'Used in' column
    - QRSwideMSE: new row for v12
  Sheet 3 (Training Progress):
    - v1 resume: add ep399 final
    - v11: add ep50-59
    - v12: add ep1 start
  Sheet 4 (Architecture Details):
    - title: include v12
    - status row: v9 DONE, v11 ep59+, v1 resume DONE
    - best val row: v1 resume 0.013776, v9 final
  Sheet 5 (Amplitude Analysis):
    - v12 row: 'Not yet implemented' -> RUNNING ep1+
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

PATH = '/shared_storage/iulia.orvas/paper/fECG_approx/experiments.xlsx'

wb = openpyxl.load_workbook(PATH)

PROTECTED = {'Loss Curves', 'Inference Plots'}
assert all(s in wb.sheetnames for s in PROTECTED), \
    f"ABORT: image sheets not found in {wb.sheetnames}"

N      = '\n'
GREEN  = PatternFill('solid', fgColor='C6EFCE')
ORANGE = PatternFill('solid', fgColor='FFEB9C')
BLUE   = PatternFill('solid', fgColor='BDD7EE')
thin   = Side(style='thin')
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP   = Alignment(wrap_text=True, vertical='top')
CENTER = Alignment(horizontal='center', vertical='top', wrap_text=True)


def sc(ws, row, col, value, fill=None, bold=False, height=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.alignment = WRAP
    cell.border    = BORDER
    if fill:   cell.fill = fill
    if bold:   cell.font = Font(bold=True)
    if height: ws.row_dimensions[row].height = height
    return cell


def row_data(ws, row, vals, fill=None, height=60):
    for c, v in enumerate(vals, 1):
        sc(ws, row, c, v, fill=fill)
    ws.row_dimensions[row].height = height


# =============================================================================
# SHEET 1 — Model Overview
# =============================================================================
ws1 = wb['Model Overview']
ws1.cell(row=1, column=1).value = \
    'fECG Extraction — Model Overview  |  updated 2026-04-27'

# v1 resume (row 4): DONE
sc(ws1, 4,  2, f'DONE{N}ep 399/400{N}2026-04-27',    fill=GREEN)
sc(ws1, 4, 11, '0.013776 @ ep 399',                   fill=GREEN)
sc(ws1, 4, 12,
   f'Best signal quality overall{N}'
   f'Amplitudes ~90-95% GT, clean QRS waveform{N}'
   f'SignalMSE only — no external interference',        fill=GREEN)

# v9 (row 12): DONE
sc(ws1, 12,  2, f'DONE{N}ep 138/300{N}2026-04-26',   fill=ORANGE)
sc(ws1, 12, 11, '0.925510 @ ep 138',                  fill=ORANGE)
sc(ws1, 12, 12,
   f'ComplexMSE dominates ~10-20x SignalMSE at convergence{N}'
   f'R-peak amplitudes suppressed vs GT{N}'
   f'Val loss 0.925 not directly comparable to v1/v11 (different scale)', fill=ORANGE)

# v11 (row 14): update progress
sc(ws1, 14,  2,
   f'RUNNING{N}ep 59+/300, job 1359{N}early stop 9/20, best @ ep49', fill=BLUE)
sc(ws1, 14, 11,
   f'0.446454 @ ep 49{N}(sig~0.047, pk~0.124){N}LR=2.5e-5 (reduced twice)', fill=BLUE)

# v12 — new row (row 15)
row_data(ws1, 15, [
    'v12',
    f'RUNNING{N}ep 1+/300, job 1371{N}2026-04-27',
    '1371', 'Lenovo2',
    f'ComplexUNetV12 (1.87M){N}Concat skip, RoActivation{N}Diag(phase), Gain mask 1.5x',
    '1.87 M',
    f'Gain mask{N}1.5 x sigmoid x mixture',
    f'SignalMSE{N}+ 3x QRSwideMSE{N}(fqrs +/-100ms)',
    '1e-4 / 16', 'Random',
    '—',
    f'mask in [0, 1.5] -> recovers energy at antiphase STFT bins{N}'
    f'QRSwideMSE: 50.2% coverage vs ~15% for +/-30ms{N}'
    f'Covers full PQRS complex (Q+R+S waves){N}'
    f'No ComplexMSE -> no amplitude suppression',
], fill=BLUE, height=60)


# =============================================================================
# SHEET 2 — Loss Functions
# =============================================================================
ws2 = wb['Loss Functions']

# PeakMSE(fqrs) (row 7, col 5 "Used in")
sc(ws2, 7, 5,
   f'v10 (3x) + ComplexMSE — INEFFECTIVE{N}'
   f'v11 (3x) without ComplexMSE — RUNNING ep59+, best=0.446{N}'
   f'v12 uses QRSwideMSE +/-100ms instead',
   fill=BLUE, height=85)

# QRSwideMSE — new row (row 9)
row_data(ws2, 9, [
    f'QRSwideMSE{N}(fqrs +/-100ms)',
    f'mask = 1 at fqrs +/-100 samples, 0 elsewhere{N}'
    f'n = mask.sum() * n_channels{N}'
    f'loss = sum((pred-target)^2 * mask) / n',
    f'Time domain{N}(binary mask, wide window)',
    f'Supervises full PQRS complex (Q, R, S waves){N}'
    f'50.2% signal coverage vs ~15% for +/-30ms{N}'
    f'Forces correct amplitudes across entire ventricular complex{N}'
    f'Not just R-peak tip',
    f'v12 (3x) without ComplexMSE — RUNNING ep1+',
], fill=BLUE, height=85)


# =============================================================================
# SHEET 3 — Training Progress
# =============================================================================
ws3 = wb['Training Progress']

# v1 resume: add ep399 final (row 27)
sc(ws3, 27, 1, 399,       fill=GREEN, height=18)
sc(ws3, 27, 3, 0.013776,  fill=GREEN)
ws3.cell(row=27, column=1).font = Font(bold=True)
ws3.cell(row=27, column=3).font = Font(bold=True)
sc(ws3, 27, 4, 'BEST FINAL — DONE 2026-04-27', fill=GREEN)
ws3.cell(row=27, column=4).font = Font(bold=True, color='006100')

# v9: overwrite row 17 (was ep155 "running") with final result
ws3.cell(row=17, column=5).value  = '138 (BEST FINAL)'
ws3.cell(row=17, column=6).value  = '0.9255 — DONE ep138 2026-04-26'
ws3.cell(row=17, column=5).font   = Font(bold=True)
ws3.cell(row=17, column=6).font   = Font(bold=True)
ws3.cell(row=17, column=5).fill   = GREEN
ws3.cell(row=17, column=6).fill   = GREEN
ws3.cell(row=17, column=5).border = BORDER
ws3.cell(row=17, column=6).border = BORDER

# v11: add ep50-59 in col 7, rows 12-16
V11_EXTRA = [
    (12, 'ep50: 0.4832 (sig=0.0585 pk=0.1416) — early stop 1/20'),
    (13, 'ep52: 0.4468 — below best, not saved'),
    (14, 'ep54: 0.4544 — early stop 5/20'),
    (15, 'ep58: 0.4602 — LR -> 2.5e-5, early stop 9/20'),
    (16, 'ep59: training in progress (~batch 1500/5220)'),
]
for r, txt in V11_EXTRA:
    c = ws3.cell(row=r, column=7, value=txt)
    c.border    = BORDER
    c.alignment = WRAP
    ws3.row_dimensions[r].height = 18

# v12: add header + ep1 data in col 9
ws3.cell(row=2, column=9).value     = 'v12 (gain mask 1.5x, QRSwideMSE +/-100ms)'
ws3.cell(row=2, column=9).font      = Font(bold=True)
ws3.cell(row=2, column=9).fill      = BLUE
ws3.cell(row=2, column=9).alignment = CENTER
ws3.cell(row=2, column=9).border    = BORDER
ws3.cell(row=3, column=9).value     = 'ep1: avg~0.41 (sig~0.09 pk~0.10) — stable start'
ws3.cell(row=3, column=9).border    = BORDER
ws3.cell(row=3, column=9).alignment = WRAP
ws3.cell(row=3, column=9).fill      = BLUE
ws3.column_dimensions['I'].width    = 40


# =============================================================================
# SHEET 4 — Architecture Details
# =============================================================================
ws4 = wb['Architecture Details']
ws4.cell(row=1, column=1).value = 'Architecture Details — v1 through v12'

# Row 12 — Best val loss  (cols: 2=v1/v5, 3=v8, 4=v9, 5=v10, 6=v11, 7=v1resume, 8=Notes)
sc(ws4, 12, 6, f'0.446454{N}ep49 (best){N}LR=2.5e-5 now',  fill=GREEN)
sc(ws4, 12, 7, f'0.013776{N}ep399 DONE',                    fill=GREEN)

# Row 13 — Status
sc(ws4, 13, 4, f'DONE{N}(ep138 2026-04-26)',                fill=BLUE)
sc(ws4, 13, 6, f'RUNNING{N}ep59+, job1359{N}early stop 9/20', fill=BLUE)
sc(ws4, 13, 7, f'DONE{N}(ep399 2026-04-27)',                fill=BLUE)

# v12 summary row (row 16, with thin spacer at row 15)
ws4.row_dimensions[15].height = 8
row_data(ws4, 16, [
    'v12 (NEW)',
    '—', '—',
    'v9 (parent arch)',
    '—',
    f'Similar to v11{N}but gain mask instead of direct pred',
    '—',
    f'v12 = v9 arch + 1.5x sigmoid mask + QRSwideMSE +/-100ms + no ComplexMSE{N}'
    f'Gain mask [0, 1.5]: can exceed mixture magnitude at antiphase STFT bins{N}'
    f'QRSwideMSE: 50.2% coverage -> full PQRS (vs 15% in v11){N}'
    f'Job 1371 Lenovo2, launched 2026-04-27',
], fill=BLUE, height=70)


# =============================================================================
# SHEET 5 — Amplitude Analysis
# =============================================================================
ws5 = wb['Amplitude Analysis']

# Proposed v12 row (row 7): update Factor (col 1), Affected models (col 3), Solution (col 4)
sc(ws5, 7, 1,
   f'v12 — RUNNING{N}Soft mask gain 1.5x{N}job 1371, Lenovo2',
   fill=BLUE)
sc(ws5, 7, 3,
   f'v12 — RUNNING ep1+{N}job 1371, Lenovo2{N}launched 2026-04-27',
   fill=BLUE)
sc(ws5, 7, 4,
   f'IMPLEMENTED AND RUNNING{N}Configuration:{N}'
   f'  mask = 1.5 x sigmoid(logits), range [0, 1.5]{N}'
   f'  loss = SignalMSE + 3x QRSwideMSE(fqrs +/-100ms){N}'
   f'  no ComplexMSE{N}'
   f'  DILATION=100 (50.2% coverage){N}'
   f'Expected: amplitudes > v1 via gain + full PQRS supervision',
   fill=BLUE)


# =============================================================================
# SAVE
# =============================================================================
wb.save(PATH)
print(f'Saved: {PATH}')
print(f'Sheets: {wb.sheetnames}')
print('Image sheets preserved:', [s for s in wb.sheetnames if s in PROTECTED])
