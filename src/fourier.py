import torch
import torchaudio


class STFT():
    def __init__(self, nfft=256, win_len=100, hop_len=15):
        super().__init__()
        self.nfft = nfft
        self.hoplen = hop_len
        self.win_len = win_len
        self.window = torch.hann_window(win_len)

  
    def stft_only(self, signal):
        return torch.stft(
            signal, 
            n_fft=self.nfft, 
            hop_length=self.hoplen, 
            win_length=self.win_len, 
            window=self.window,
            pad_mode='constant',
            normalized=False,
            return_complex=True
        )
        
    def istft_only(self, spec, length=None):
        return torch.istft(
            spec,
            n_fft=self.nfft, 
            hop_length=self.hoplen, 
            win_length=self.win_len,
            window=self.window,
            center=True,
            length=length
        )
    
    def stft(self, signal):
        spec = self.stft_only(signal)
        
        ref = torch.max(torch.abs(spec))
        spec_db = torchaudio.functional.amplitude_to_DB(
            torch.abs(spec), 
            multiplier=20, 
            amin=1e-05, 
            db_multiplier=torch.log10(ref),
            top_db=80
        )
        phase = torch.angle(spec)
        
        # drop last freq bin
        spec_db = spec_db[:, :-1]
        phase = phase[:, :-1]
        
        spec_db = (spec_db + 80) / 80  # scale to [0, 1]
        return spec_db, phase, ref
    
        
    def istft(self, spec_db, phase=None, length=None, ref=1):
        spec_db = spec_db * 80 - 80  # scale to [-80, 0]

        # add last freq bin
        b, f, t = spec_db.shape
        spec_db = torch.cat([spec_db, torch.zeros(b, 1, t, dtype=spec_db.dtype, device=spec_db.device)], dim=1)

        spec = torchaudio.functional.DB_to_amplitude(spec_db, ref=ref, power=0.5)
        if phase is not None:
            phase = torch.cat([phase, torch.zeros(b, 1, t, dtype=phase.dtype, device=phase.device)], dim=1)
            spec = spec * torch.exp(1j * phase)
        
        signal = self.istft_only(spec, length=length)
        return signal
        
    
    def stft_batched(self, signals):
        specs, phases, refs = [], [], []
        for signal in signals:
            spec, phase, ref = self.stft(signal)
            specs.append(spec)
            phases.append(phase)
            refs.append(ref)
        return torch.stack(specs), torch.stack(phases), torch.stack(refs)
    
    
    def istft_batched(self, specs, phases, refs, lengths):
        out = []
        for spec, length in zip(specs, lengths):
            signal = self.istft(spec, length=length)
            out.append(signal)
        return torch.stack(out)
    
    
    

### Tests to check if the implementation is consistent with librosa
if __name__ == '__main__':
    import numpy as np
    import librosa


    def stft(signal, nfft, hoplen, win_len):
        spec = librosa.stft(signal, n_fft=nfft, hop_length=hoplen, win_length=win_len)
        ref = np.max(np.abs(spec))
        spec_db = librosa.amplitude_to_db(np.abs(spec), ref=ref)
        phase = np.angle(spec)
        
        spec_db = spec_db[:, :-1]
        phase = phase[:, :-1]
        spec_db = (spec_db + 80.0) / 80.0  # scale to [0, 1]
        return spec_db, phase, ref


    def istft(spec_db, nfft, hoplen, win_len, phase=None, length=None, ref=1):
        spec_db = spec_db * 80 - 80
        b, f, t = spec_db.shape
        spec_db = np.concatenate([spec_db, np.zeros((b, 1, t))], axis=1)
        
        spec = librosa.db_to_amplitude(spec_db, ref=ref)
        if phase is not None:
            phase = np.concatenate([phase, np.zeros((b, 1, t))], axis=1)
            spec = spec * np.exp(1j * phase)
        signal = librosa.istft(spec, n_fft=nfft, hop_length=hoplen, win_length=win_len, length=length)
        return signal
    
    
    signal = np.random.randn(32, 50000)
    signal /= np.max(np.abs(signal))
    nfft = 256
    win_len = 100
    hoplen = 15


    stft_obj = STFT(nfft=nfft, win_len=win_len, hop_len=hoplen)
    
    spec_db, phase, ref = stft_obj.stft(torch.tensor(signal))
    spec_db1, phase1, ref1 = stft(signal, nfft=nfft, hoplen=hoplen, win_len=win_len)
    
    
    print(f'spec_db: [{spec_db.min().item()}, {spec_db.max().item()}]', f'spec_db1: [{spec_db1.min()}, {spec_db1.max()}]')
    
    
    print(np.allclose(spec_db.numpy(), spec_db1, atol=1e-3))
    print(np.allclose(phase.numpy(), phase1, atol=1e-3))
    print(ref.item(), ref1)
    
    
    rec = stft_obj.istft(spec_db, phase, ref=ref)
    rec1 = istft(spec_db1, nfft=nfft, hoplen=hoplen, win_len=win_len, phase=phase1, ref=ref1)
    
    print(np.allclose(rec.numpy(), rec1, atol=1e-3))
    print(np.max(np.abs(rec.numpy() - rec1)))  # around e-8
    