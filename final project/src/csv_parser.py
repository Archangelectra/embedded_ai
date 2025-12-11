import pandas as pd
import matplotlib.pyplot as plt
import os

# CONFIG
RESULTS_PATH = "../results/results.csv"

def plot_training_history(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    # 1. load Data
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. select phase changes
    data['global_epoch'] = data.index + 1
    
    # find indices where the epoch number decreases (indicating a new phase)
    phase_changes = data[data['epoch'].diff() < 0].index.tolist()

    # 3. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- PLOT 1: LOSS (MSE) ---
    ax1.plot(data['global_epoch'], data['loss'], label='Train Loss (MSE)', color='blue', linewidth=2)
    if 'val_loss' in data.columns:
        ax1.plot(data['global_epoch'], data['val_loss'], label='Val Loss (MSE)', color='orange', linestyle='--', linewidth=2)
    
    ax1.set_ylabel('Loss (Mean Squared Error)')
    ax1.set_title('Model Loss (Lower is Better)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- PLOT 2: METRIC (MAE) ---
    ax2.plot(data['global_epoch'], data['mae'], label='Train MAE', color='green', linewidth=2)
    if 'val_mae' in data.columns:
        ax2.plot(data['global_epoch'], data['val_mae'], label='Val MAE', color='red', linestyle='--', linewidth=2)
    
    ax2.set_ylabel('Mean Absolute Error (Normalized)')
    ax2.set_xlabel('Total Epochs')
    ax2.set_title('Model Accuracy (Lower is Better)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 4. mark phases
    for i, idx in enumerate(phase_changes):
        # adjust x-coordinate to align with global_epoch
        x_coord = data.loc[idx, 'global_epoch'] - 0.5 
        
        # draw Line on both plots
        ax1.axvline(x=x_coord, color='black', linestyle=':', alpha=0.7)
        ax2.axvline(x=x_coord, color='black', linestyle=':', alpha=0.7)
        
        # add label
        ax1.text(x_coord, ax1.get_ylim()[1]*0.9, f' Phase {i+2} Start ', rotation=90, verticalalignment='top')

    plt.tight_layout()
    
    # 5. save or show
    output_img = "../results/training_graph.png"
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_training_history(RESULTS_PATH)