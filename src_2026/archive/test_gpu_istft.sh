#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=Lenovo2
#SBATCH --output=/shared_storage/iulia.orvas/paper/fECG_approx/logs/test_gpu_istft.log

export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH
source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

python3 -c "
import torch, time
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
device = 'cuda'

# Test GPU iSTFT
B, F, T = 192, 129, 401   # B*C = 32*6
spec_gpu = torch.randn(B, F, T, dtype=torch.cfloat, device=device, requires_grad=True)
win_gpu = torch.hann_window(128, device=device)
try:
    t0 = time.time()
    out = torch.istft(spec_gpu, n_fft=256, hop_length=10, win_length=128, window=win_gpu, length=4000)
    out.sum().backward()
    print(f'GPU istft OK  shape={out.shape}  time={time.time()-t0:.3f}s')
except Exception as e:
    print('GPU istft FAILED:', e)

# Compare with CPU
spec_cpu = spec_gpu.detach().cpu().requires_grad_(True)
win_cpu = torch.hann_window(128)
t0 = time.time()
out_cpu = torch.istft(spec_cpu, n_fft=256, hop_length=10, win_length=128, window=win_cpu, length=4000)
out_cpu.sum().backward()
print(f'CPU istft OK  shape={out_cpu.shape}  time={time.time()-t0:.3f}s')
"
