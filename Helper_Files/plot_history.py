import matplotlib.pyplot as plt
import numpy as np
import config
import os


def plot_training_curve(path):
   
    
    G1_train_loss = np.loadtxt(f"{path}/G1_train_loss_{name}.csv")
    G2_train_loss = np.loadtxt(f"{path}/G2_train_loss_{name}.csv")
    D_train_loss = np.loadtxt(f"{path}/D_train_loss_{name}.csv")
    G1_Val_loss = np.loadtxt(f"{path}/G1_Val_loss_{name}.csv")
    G2_Val_loss = np.loadtxt(f"{path}/G2_Val_loss_{name}.csv")
    D_Val_loss = np.loadtxt(f"{path}/D_Val_loss_{name}.csv")
    
    n = len(G1_train_loss)

    plt.subplot(2, 2, 1)
    plt.plot(range(1, n + 1), G1_train_loss, label="G1 Train")
    plt.plot(range(1, n + 1), G1_Val_loss, label="G1 Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.title("G1 Train vs Val Loss")

    plt.subplot(2, 2, 2)
    plt.plot(range(1, n + 1), G2_train_loss, label="G2 Train")
    plt.plot(range(1, n + 1), G2_Val_loss, label="G2 Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.title("G2 Train vs Val Loss")

    plt.subplot(2, 2, 3)
    plt.plot(range(1, n+ 1), D_train_loss, label="D Train")
    plt.plot(range(1, n + 1), D_Val_loss, label="D Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.title("D Train vs Val Loss")
    
    
    G_total_train_loss = np.add(G1_train_loss, G2_train_loss)
    G_total_val_loss = np.add(G1_Val_loss, G2_Val_loss)
    plt.subplot(2, 2, 4)
    plt.plot(range(1, n + 1), G_total_train_loss, label="Total G Train")
    plt.plot(range(1, n + 1), G_total_val_loss, label="Total G Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.title("Generator Train vs Val Loss")

    plt.tight_layout()
    
    plot_filename = os.path.join("History_plots/", f"{name}_curves.png")
    plt.savefig(plot_filename)
    plt.show()
    


name = config.NAME
plot_training_curve("History")



