"""Quick script to inspect test .mat file structure."""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def inspect_mat_file(mat_path):
    """Inspect structure and dimensions of .mat file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {mat_path}")
    print('='*60)
    
    data = sio.loadmat(mat_path)
    
    # Print top-level keys
    print(f"\nTop-level keys: {list(data.keys())}")
    
    # Handle 'out' structure
    if 'out' in data:
        out_data = data['out'][0, 0]
        print(f"\nInside 'out' object:")
        print(f"  Type: {type(out_data)}")
        
        if hasattr(out_data, 'dtype'):
            print(f"  Fields: {out_data.dtype.names}")
            
            for field in out_data.dtype.names:
                value = out_data[field]
                print(f"\n  {field}:")
                print(f"    Type: {type(value)}")
                print(f"    Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                
                # If nested structure (1,1) containing array, extract it
                if hasattr(value, 'shape') and value.shape == (1, 1) and value.dtype == object:
                    nested = value[0, 0]
                    print(f"    Nested object type: {type(nested)}")
                    if hasattr(nested, 'shape'):
                        print(f"    Nested shape: {nested.shape}")
                        if nested.ndim > 0 and nested.size > 0:
                            print(f"    Nested dtype: {nested.dtype}")
                            try:
                                print(f"    Min: {nested.min():.6f}, Max: {nested.max():.6f}")
                                print(f"    Mean: {nested.mean():.6f}, Std: {nested.std():.6f}")
                            except (TypeError, ValueError):
                                print(f"    (Cannot compute min/max for this dtype)")
                # Skip object arrays that aren't simple nested structures
                elif hasattr(value, 'dtype') and value.dtype == object:
                    print(f"    (Object array - skipping stats)")
                # Regular numeric array
                elif hasattr(value, 'shape') and value.size > 0:
                    print(f"    Data type: {value.dtype}")
                    try:
                        print(f"    Min: {value.min():.6f}, Max: {value.max():.6f}")
                        print(f"    Mean: {value.mean():.6f}, Std: {value.std():.6f}")
                    except (TypeError, ValueError):
                        print(f"    (Cannot compute min/max for this dtype)")
        
        # Plot - only channel 0
        fig, axes = plt.subplots(3, 1, figsize=(14, 8))
        
        # Extract channel 0 - consistent extraction
        mecg_ch0 = out_data['mecg'][0, :]
        
        # fecg is nested (1,1) -> (6, 600000), so extract [0,0] first
        # fecg_data = out_data['fecg'][0, 0]  # Now (6, 600000)
        # fecg_ch0 = fecg_data[0, :]
        fecg_ch0 = out_data['fecg'][0, :]

        if 'mixture' in out_data.dtype.names:
            mixture_ch0 = out_data['mixture'][0, :]
        else:            
            print(f"\nNo 'mixture' field found in 'out'.")
            mixture_ch0 = mecg_ch0 + fecg_ch0
        
        time_axis = np.arange(len(mixture_ch0)) / 1000  # seconds
        
        # Plot first 1000 samples for visibility
        axes[0].plot(time_axis[0:10000], mixture_ch0[0:10000], label='Mixture from .mat', color='blue', alpha=0.7, linewidth=1)
        axes[0].plot(time_axis[0:10000], fecg_ch0[0:10000] + mecg_ch0[0:10000], label='Mixture = mECG + fECG', color='green', alpha=0.5, linewidth=1)
        axes[0].set_title("Channel 0: Input (Mixture)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_axis[0:10000], fecg_ch0[0:10000], label='Ground Truth fECG', color='green', alpha=0.7, linewidth=1)
        axes[1].set_title("Channel 0: fECG")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time_axis[0:10000], mecg_ch0[0:10000], label='mECG', color='orange', alpha=0.7, linewidth=1)
        axes[2].set_title("Channel 0: mECG")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_data_inspection_' + test_file + '.png', dpi=150)
        print(f"\nPlot saved to: test_data_inspection_" + test_file + ".png")
        plt.close()

    elif 'ecg' in data:
        print(f"\nFound 'ecg' key with shape: {data['ecg'].shape}")
    else:
        print(f"\nNo 'out' or 'ecg' key found in .mat file.")


if __name__ == '__main__':
    import sys
    
    test_file = 'fecgsyn01.mat'

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    inspect_mat_file(test_file)
