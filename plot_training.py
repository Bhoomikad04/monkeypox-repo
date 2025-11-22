# plot_training.py
import matplotlib.pyplot as plt
import os

def plot_and_save_history(history_dict, model_name, out_dir="training_plots"):
    """
    history_dict: dictionary like history.history returned by model.fit()
    model_name: string used for filename
    out_dir: output folder
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    if "accuracy" in history_dict:
        plt.plot(epochs, history_dict["accuracy"], label="Train")
    if "val_accuracy" in history_dict:
        plt.plot(epochs, history_dict["val_accuracy"], label="Val")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, history_dict.get("loss", []), label="Train")
    plt.plot(epochs, history_dict.get("val_loss", []), label="Val")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_training_plot.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved plot:", out_path)
