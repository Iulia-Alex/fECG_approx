"""
Actualizează experiments.xlsx cu rezultatele binary_v1.
- Part2 Classification: adaugă secțiune Binary Detection + rând binary_v1
- Inference Plots: inserează plotul de inferență binary_v1
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.drawing.image import Image as XLImage

PATH      = '/shared_storage/iulia.orvas/paper/fECG_approx/experiments.xlsx'
PLOT_PATH = '/shared_storage/iulia.orvas/paper/fECG_approx/plots/binary_v1_inference_summary.png'

wb = openpyxl.load_workbook(PATH)

thin   = Side(style='thin')
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP   = Alignment(wrap_text=True, vertical='top')
CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)
N      = '\n'

GREEN  = PatternFill('solid', fgColor='C6EFCE')
BLUE   = PatternFill('solid', fgColor='BDD7EE')
PURPLE = PatternFill('solid', fgColor='E2CFEA')
ORANGE = PatternFill('solid', fgColor='FFEB9C')
GRAY   = PatternFill('solid', fgColor='D9D9D9')


def sc(ws, row, col, value, fill=None, bold=False, italic=False, height=None, align=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.alignment = align or WRAP
    cell.border    = BORDER
    if fill:   cell.fill = fill
    font_kwargs = {}
    if bold:   font_kwargs['bold'] = True
    if italic: font_kwargs['italic'] = True
    if font_kwargs: cell.font = Font(**font_kwargs)
    if height: ws.row_dimensions[row].height = height
    return cell


# ==========================================================================
# SHEET: Part2 Classification — adaugă secțiune Binary Detection
# ==========================================================================
ws2 = wb['Part2 Classification']

# Rândul curent max
last_row = ws2.max_row + 2   # 2 rânduri spațiu

# Separator + titlu secțiune
sc(ws2, last_row, 1,
   'Part 2b — Binary Movement Detection (movement / no-movement)',
   fill=PURPLE, bold=True, height=22, align=CENTER)
for col in range(2, 7):
    c = ws2.cell(row=last_row, column=col)
    c.fill   = PURPLE
    c.border = BORDER

ws2.merge_cells(
    start_row=last_row, start_column=1,
    end_row=last_row,   end_column=6
)

# Header rând
hdr_row = last_row + 1
headers = ['Model', 'Input', 'Architecture', 'F1 / Acc (test)', 'Note', 'Status']
for col, h in enumerate(headers, 1):
    sc(ws2, hdr_row, col, h, fill=GRAY, bold=True, height=18)

# binary_v1 rând
data_row = hdr_row + 1
sc(ws2, data_row, 1, 'binary_v1', fill=BLUE, bold=True, height=80)
sc(ws2, data_row, 2,
   f'8 canale per fereastră 8s:{N}'
   f'  0-5: fECG z-norm (6 canale){N}'
   f'  6: A_QRS (amplitudine QRS per bătaie, Rooijakkers 2016){N}'
   f'  7: Baseline wander (mov avg 1s)',
   fill=BLUE)
sc(ws2, data_row, 3,
   f'PrecisionResUNet (5.99M){N}'
   f'4 enc/dec ResidualBlocks (Conv1d k=15, GroupNorm){N}'
   f'ASPP bottleneck (dilații 1,2,4,8 + global avg pool){N}'
   f'TransformerBottleneck (2 layers, 4 heads){N}'
   f'Ieșire: mască binară per sample',
   fill=BLUE)
sc(ws2, data_row, 4,
   f'F1  = 0.921 (medie 6 fișiere test){N}'
   f'Acc = 0.930 (medie 6 fișiere test){N}'
   f'(Short_time_intervals, fișiere nevăzute){N}'
   f'@ epoch 12/100',
   fill=BLUE, bold=True)
sc(ws2, data_row, 5,
   f'Fără informație despre granițele GT la inferență{N}'
   f'Prezice mască continuă per-sample pe semnal brut{N}'
   f'Loss: BCE + Dice (pos_weight=1.5){N}'
   f'Stride train=4s, val=8s (non-overlap){N}'
   f'File-level split: 557 train / 98 val (15%){N}'
   f'Antrenat pe Long_time_intervals (654 fișiere)',
   fill=BLUE)
sc(ws2, data_row, 6,
   f'RUNNING{N}epoch 12+/100{N}job 1386 Lenovo2{N}2026-05-07',
   fill=ORANGE)

# Ajustare lățimi coloane
ws2.column_dimensions['A'].width = 14
ws2.column_dimensions['B'].width = 38
ws2.column_dimensions['C'].width = 35
ws2.column_dimensions['D'].width = 22
ws2.column_dimensions['E'].width = 42
ws2.column_dimensions['F'].width = 16


# ==========================================================================
# SHEET: Inference Plots — adaugă plotul binary_v1
# ==========================================================================
ws_plots = wb['Inference Plots']

# Găsim ultimul rând folosit
last_img_row = ws_plots.max_row + 3

# Titlu
sc(ws_plots, last_img_row, 1,
   f'binary_v1 — Inferență test set (Short_time_intervals){N}'
   f'Epoch 12/100  |  Mean F1=0.921  |  Mean Acc=0.930{N}'
   f'3 fișiere × 60s preview  |  2026-05-07',
   fill=PURPLE, bold=True, height=40)

# Inserare imagine
try:
    img = XLImage(PLOT_PATH)
    img.width  = 900
    img.height = 450
    cell_anchor = ws_plots.cell(row=last_img_row + 1, column=1).coordinate
    ws_plots.add_image(img, cell_anchor)
    print('Plot inserat în Inference Plots.')
except Exception as e:
    print(f'WARN: nu am putut insera imaginea: {e}')


# ==========================================================================
# SAVE
# ==========================================================================
wb.save(PATH)
print(f'Salvat: {PATH}')
print(f'Sheets: {wb.sheetnames}')
