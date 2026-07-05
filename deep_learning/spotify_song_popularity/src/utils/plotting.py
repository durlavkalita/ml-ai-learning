import matplotlib.pyplot as plt

def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
) -> None:

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.show()
    # plt.savefig("loss_curve.png")