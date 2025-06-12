import matplotlib.pyplot as plt

def plot_curves(train_losses, val_scores):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_scores, label="Val Dice")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()
