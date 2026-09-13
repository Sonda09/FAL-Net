import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
# Example:
# python plot_det_evolution.py \
#     --dir2015CM results_CrossMatch/ablation/proposed_r8 \
#     --dir2015HS results_Hi_Scan/ablation/proposed_r8 \
#     --dir2015DP results_Digital_Persona/ablation/proposed_r8 \
#     --dir2015GB results_GreenBit/ablation/proposed_r8
def plot_longitudinal_det(base_dirs, labels):
    plt.figure(figsize=(10, 8))
    
    found_data = False
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for dir_path, label, color in zip(base_dirs, labels, colors):
        roc_path = os.path.join(dir_path, 'roc_data.npy')
        
        if os.path.exists(roc_path):
            try:
                data = np.load(roc_path, allow_pickle=True).item()
                fpr = data['fpr']
                tpr = data['tpr']
                fnr = 1 - tpr
                
                # Filter limits for log scale
                valid_idx = (fpr > 0) & (fnr > 0)
                fpr = fpr[valid_idx]
                fnr = fnr[valid_idx]
                
                plt.plot(fpr * 100, fnr * 100, label=label, linewidth=2.5, color=color)
                found_data = True
            except Exception as e:
                print(f"Error loading {roc_path}: {e}")
        else:
            print(f"Warning: {roc_path} not found. Please ensure the model was evaluated on {label} after the recent code update.")

    if not found_data:
        print("\nNo roc_data.npy files found in any specified directories. Please run your evaluation scripts first.")
        return

    # Add reference EER line (y = x)
    plt.plot([0.01, 100], [0.01, 100], 'k--', label='EER', alpha=0.5)

    # Formal Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", color='0.9')
    plt.xlabel('APCER %')
    plt.ylabel('BPCER %')
    #plt.title('Performance Evolution Across LivDet 2015 sensors (DET Curve)')
    plt.legend(loc='upper right')
    
    # Biometric industry limits
    plt.xlim(0.01, 100)
    plt.ylim(0.01, 100)
    
    output_png = 'det_evolution_livdet.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nEvolution DET curve saved to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Longitudinal Biometric DET curve for LivDet datasets')
    # Use default likely paths where users would run test.py for different datasets
    parser.add_argument('--dir2015CM', type=str, default='results_CrossMatch/ablation/proposed_r8', help='Output dir for 2015 CrossMatch')
    parser.add_argument('--dir2015HS', type=str, default='results_Hi_Scan/ablation/proposed_r8', help='Output dir for 2015 Hi_Scan')
    parser.add_argument('--dir2015DP', type=str, default='results_Digital_Persona/ablation/proposed_r8', help='Output dir for 2015 Digital_Persona')
    parser.add_argument('--dir2015GB', type=str, default='results_GreenBit/ablation/proposed_r8', help='Output dir for 2015 GreenBit')



    args = parser.parse_args()
    
    directories = [args.dir2015CM, args.dir2015HS, args.dir2015DP, args.dir2015GB]
    labels = ['CrossMatch', 'Hi_Scan', 'Digital_Persona', 'GreenBit']
    
    plot_longitudinal_det(directories, labels)
