import scipy.io, glob, numpy as np

files = sorted(glob.glob('../data/movement_ecg/*.mat'))
results = []

for f in files[:15]:
    try:
        mat = scipy.io.loadmat(f, simplify_cells=True)
        out = mat['out']
        fecg    = out['fecg'].astype(float)
        mecg    = out['mecg'].astype(float)
        mixture = out['mixture'].astype(float)
        noise_raw = out['noise']
        # noise is (2,) object array: [movement_noise, other_noise] each (6,N)
        if noise_raw.dtype == object:
            noise = np.vstack([np.array(n, dtype=float) for n in noise_raw])
        else:
            noise = noise_raw.astype(float)

        p_fecg    = np.mean(fecg**2)
        p_mecg    = np.mean(mecg**2)
        p_mixture = np.mean(mixture**2)
        p_noise   = np.mean(noise**2)

        snr_fecg_mecg  = 10*np.log10(p_fecg / p_mecg)
        snr_fecg_mix   = 10*np.log10(p_fecg / p_mixture)
        snr_fecg_noise = 10*np.log10(p_fecg / p_noise)
        snr_mecg_noise = 10*np.log10(p_mecg / p_noise)

        name = f.split('/')[-1].replace('fecgsyn_Long_time_segment_','').replace('.mat','')
        results.append([snr_fecg_mecg, snr_fecg_mix, snr_fecg_noise, snr_mecg_noise])
        print(f'{name[:30]:30s}  fecg/mecg={snr_fecg_mecg:+6.1f}dB  fecg/mix={snr_fecg_mix:+6.1f}dB  fecg/noise={snr_fecg_noise:+6.1f}dB  mecg/noise={snr_mecg_noise:+6.1f}dB')
    except Exception as e:
        print(f.split('/')[-1][:50], '-> SKIP:', e)

if results:
    arr = np.array(results)
    print()
    print(f'MEAN: fecg/mecg={arr[:,0].mean():+.1f}dB  fecg/mix={arr[:,1].mean():+.1f}dB  fecg/noise={arr[:,2].mean():+.1f}dB  mecg/noise={arr[:,3].mean():+.1f}dB')
    print(f'STD:  fecg/mecg={arr[:,0].std():.1f}dB   fecg/mix={arr[:,1].std():.1f}dB    fecg/noise={arr[:,2].std():.1f}dB    mecg/noise={arr[:,3].std():.1f}dB')
