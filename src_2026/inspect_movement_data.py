import scipy.io as sio
import numpy as np
import os

data_dir = "../data/movement_ecg"
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])

fpath = os.path.join(data_dir, files[0])
mat = sio.loadmat(fpath)
out = mat['out']

# Inspect fecg deeper
print("=== fecg deep inspection ===")
fecg_raw = out['fecg'][0][0]
print(f"fecg_raw type={type(fecg_raw)}, shape={fecg_raw.shape}, dtype={fecg_raw.dtype}")
if fecg_raw.dtype == object:
    inner = fecg_raw.flat[0]
    print(f"  inner type={type(inner)}, shape={getattr(inner,'shape','N/A')}, dtype={getattr(inner,'dtype','N/A')}")
    if hasattr(inner, 'shape') and np.issubdtype(inner.dtype, np.number):
        print(f"  min={inner.min():.4f}, max={inner.max():.4f}")
else:
    print(f"  shape={fecg_raw.shape}, min={fecg_raw.min():.4f}, max={fecg_raw.max():.4f}")

# Inspect fqrs deeper
print("\n=== fqrs deep inspection ===")
fqrs_raw = out['fqrs'][0][0]
print(f"fqrs_raw type={type(fqrs_raw)}, shape={fqrs_raw.shape}, dtype={fqrs_raw.dtype}")
if fqrs_raw.dtype == object:
    inner = fqrs_raw.flat[0]
    print(f"  inner: shape={getattr(inner,'shape','N/A')}, dtype={getattr(inner,'dtype','N/A')}")
    print(f"  first 10 values: {inner.flat[:10]}")

# Check param.fs
print("\n=== param.fs ===")
param = out['param'][0][0]
fs = param['fs'][0][0][0][0]
print(f"Sampling rate fs={fs} Hz")
n = param['n'][0][0][0][0]
print(f"n (total samples)={n}")
duration = n / fs
print(f"Duration = {duration:.1f} seconds")

# Confirm mixture shape
mixture = out['mixture'][0][0]
mecg = out['mecg'][0][0]
print(f"\n=== Signal shapes ===")
print(f"mixture: {mixture.shape}")
print(f"mecg:    {mecg.shape}")
print(f"synthetic_aecg: {mat['synthetic_aecg'].shape}")
print(f"\nAre mixture and synthetic_aecg the same? {np.allclose(mixture, mat['synthetic_aecg'])}")

# Check a second file to confirm consistency
fpath2 = os.path.join(data_dir, files[10])
mat2 = sio.loadmat(fpath2)
out2 = mat2['out']
mixture2 = out2['mixture'][0][0]
fecg2_raw = out2['fecg'][0][0]
print(f"\n=== Second file: {files[10]} ===")
print(f"mixture: {mixture2.shape}")
fecg2_inner = fecg2_raw.flat[0] if fecg2_raw.dtype == object else fecg2_raw
print(f"fecg: {fecg2_inner.shape}")
