import os
import numpy as np
import matplotlib.pyplot as plt
import config  # Assuming this is where you get 'name' from



def plot_training_curve_base(path):
    name = config.NAME  # Moved inside the function for clarity
    
    G1_train_loss = np.loadtxt(f"{path}/G1_train_loss_{name}.csv")
    D_train_loss = np.loadtxt(f"{path}/D_train_loss_{name}.csv")
    G1_Val_loss = np.loadtxt(f"{path}/G1_Val_loss_{name}.csv")
    D_Val_loss = np.loadtxt(f"{path}/D_Val_loss_{name}.csv")
    
    
    plt.figure(figsize=(6, 3))
    
    G_train_loss = G1_train_loss
    G_Val_loss = G1_Val_loss

    n = len(G_train_loss)

    plt.subplot(1, 2, 1)
    plt.plot(range(1, n + 1), G_train_loss, label="G Train")
    plt.plot(range(1, n + 1), G_Val_loss, label="G Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.title("Generator Train vs Val Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n + 1), D_train_loss, label="D Train")
    plt.plot(range(1, n + 1), D_Val_loss, label="D Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.title("Discriminator Train vs Val Loss")

    plt.tight_layout()
    
    plot_filename = os.path.join("History_plots", f"{name}_curves.png")
    plt.savefig(plot_filename)
    plt.show()

plot_training_curve_base("History")