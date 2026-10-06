import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_det_curves(base_dir="results_CrossMatch/ablation"):
    plt.figure(figsize=(10, 8))
    
    # We use a logarithmic scale to approximate the DET space
    # (Or formal scipy.stats.norm.ppf for proper DET scaling if needed, but log-log is very common)
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return
        
    configs = os.listdir(base_dir)
    found_data = False
    
    for config in configs:
        roc_path = os.path.join(base_dir, config, 'roc_data.npy')
        if os.path.exists(roc_path):
            try:
                data = np.load(roc_path, allow_pickle=True).item()
                fpr = data['fpr']
                tpr = data['tpr']
                fnr = 1 - tpr
                
                # Filter out pure 0s to avoid mathematical domain errors in log scale
                valid_idx = (fpr > 0) & (fnr > 0)
                fpr = fpr[valid_idx]
                fnr = fnr[valid_idx]
                
                # Plot FMR (FPR) vs FNMR (FNR)
                plt.plot(fnr * 100, fpr * 100, label=config, linewidth=2)
                found_data = True
            except Exception as e:
                print(f"Error loading {roc_path}: {e}")

    if not found_data:
        print("No roc_data.npy found. Please run the ablation script first.")
        return

    # Add reference EER line (y = x)
    plt.plot([0.01, 100], [0.01, 100], 'k--', label='EER Boundary (FMR = FNMR)', alpha=0.5)

    # Formal Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", color='0.9')
    plt.xlabel('False Match Rate (APCER) %')
    plt.ylabel('False Non-Match Rate (BPCER) %')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.legend(loc='upper right')
    
    # Set useful limits for typical biometric performance
    plt.xlim(0.05, 50)
    plt.ylim(0.05, 50)
    
    plt.savefig('det_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("DET curve saved to det_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Biometric DET curve')
    parser.add_argument('--dir', type=str, default='results_CrossMatch/ablation', help='Directory containing the configs with roc_data.npy')
    args = parser.parse_args()
    
    plot_det_curves(args.dir)
