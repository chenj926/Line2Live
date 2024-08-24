import torch
import config
from torchvision.utils import save_image

import os
import numpy as np
import matplotlib.pyplot as plt

name = config.NAME

#save generated examples to folder for visualization
def save_some_examples(gen, loader, epoch, folder):
    x, y = next(iter(loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.jpg")
        
        #save input and label once
        #if epoch == 0:
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.jpg")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.jpg")
    gen.train()





def save_gray_and_color_examples(gen1, gen2, loader, epoch, folder):
    x, y, y_gray = next(iter(loader))
    x, y, y_gray = x.to(config.DEVICE), y.to(config.DEVICE), y_gray.to(config.DEVICE)
    
    gen1.eval()
    gen2.eval()
    with torch.no_grad():
        y_fake_gray = gen1(x)
        y_fake_color = gen2(y_fake_gray)
        y_fake_gray = y_fake_gray * 0.5 + 0.5  # remove normalization
        y_fake_color = y_fake_color * 0.5 + 0.5  # remove normalization
        save_image(y_fake_color, folder + f"/y_genColor_{epoch}.jpg")
        save_image(y_fake_gray, folder + f"/y_genGray_{epoch}.jpg")
        
        #save input and label once
        #if epoch == 0:
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.jpg")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.jpg")
    gen1.train()
    gen2.train()
    

def save_triple_examples(gen1, gen2, gen3, loader, epoch, folder):
    x, y, y_gray = next(iter(loader))
    x, y, y_gray = x.to(config.DEVICE), y.to(config.DEVICE), y_gray.to(config.DEVICE)
    
    gen1.eval()
    gen2.eval()
    gen3.eval()
    with torch.no_grad():
        y_fake_gray = gen1(x)
        y_fake_color = gen2(y_fake_gray)
        y_fake_color2 = gen3(y_fake_color)
        y_fake_gray = y_fake_gray * 0.5 + 0.5  # remove normalization
        y_fake_color = y_fake_color * 0.5 + 0.5  # remove normalization
        y_fake_color2 = y_fake_color2 * 0.5 + 0.5  # remove normalization
        save_image(y_fake_color, folder + f"/y_genColor_{epoch}.jpg")
        save_image(y_fake_gray, folder + f"/y_genGray_{epoch}.jpg")
        save_image(y_fake_color2, folder + f"/y_genColorTWO_{epoch}.jpg")
        
        #save input and label once
        #if epoch == 0:
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.jpg")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.jpg")
    gen1.train()
    gen2.train()
    gen3.train()




def final_save_all(gen1, gen2, test_b1_loader, folderName):
    
    gen1.eval()
    gen2.eval()
    with torch.no_grad():
        for idx, (x, y, y_gray) in enumerate(test_b1_loader):
            x = x.to(config.DEVICE)
            y_gray = y_gray.to(config.DEVICE)
            
            y_fake_gray = gen1(x)
            y_fake_color = gen2(y_fake_gray)
            y_fake_gray = y_fake_gray * 0.5 + 0.5  # remove normalization
            y_fake_color = y_fake_color * 0.5 + 0.5  # remove normalization
            save_image(y_fake_color, folderName + f"/genColor_{idx}.jpg")
            save_image(y_fake_gray, folderName + f"/genGray_{idx}.jpg")
            
            #save input and label once
            #if epoch == 0:
            save_image(y * 0.5 + 0.5, folderName + f"/label_{idx}.jpg")
            save_image(x * 0.5 + 0.5, folderName + f"/input_{idx}.jpg")
    gen1.train()
    gen2.train()
    
def final_save_all_Triple(gen1, gen2,gen3, test_b1_loader, folderName):
    
    gen1.eval()
    gen2.eval()
    gen3.eval()
    with torch.no_grad():
        for idx, (x, y, y_gray) in enumerate(test_b1_loader):
            x = x.to(config.DEVICE)
            y_gray = y_gray.to(config.DEVICE)
            
            y_fake_gray = gen1(x)
            y_fake_color = gen2(y_fake_gray)
            y_fake_color2 = gen3(y_fake_color)
            y_fake_gray = y_fake_gray * 0.5 + 0.5  # remove normalization
            y_fake_color = y_fake_color * 0.5 + 0.5  # remove normalization
            y_fake_color2 = y_fake_color2 * 0.5 + 0.5  # remove normalization
            save_image(y_fake_color, folderName + f"/genColor_{idx}.jpg")
            save_image(y_fake_gray, folderName + f"/genGray_{idx}.jpg")
            save_image(y_fake_color2, folderName + f"/genColorTWO_{idx}.jpg")
            
            #save input and label once
            #if epoch == 0:
            save_image(y * 0.5 + 0.5, folderName + f"/label_{idx}.jpg")
            save_image(x * 0.5 + 0.5, folderName + f"/input_{idx}.jpg")
    gen1.train()
    gen2.train()
    gen3.train()
    

def final_save_all_ColorOnly(gen1, test_b1_loader, folderName):
    
    gen1.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_b1_loader):
            x = x.to(config.DEVICE)
            
            y_fake = gen1(x)
            y_fake_color = y_fake * 0.5 + 0.5  # remove normalization
            save_image(y_fake_color, folderName + f"/genColor_{idx}.jpg")
            
            
            #save input and label once
            #if epoch == 0:
            save_image(y * 0.5 + 0.5, folderName + f"/label_{idx}.jpg")
            save_image(x * 0.5 + 0.5, folderName + f"/input_{idx}.jpg")
    gen1.train()
    




#save model checkpoint 
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
    #plt.show()
    


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
    #plt.show()
    

def plot_training_curve_Triple(path):
   
    
    G1_train_loss = np.loadtxt(f"{path}/G1_train_loss_{name}.csv")
    G2_train_loss = np.loadtxt(f"{path}/G2_train_loss_{name}.csv")
    G3_train_loss = np.loadtxt(f"{path}/G3_train_loss_{name}.csv")
    D_train_loss = np.loadtxt(f"{path}/D_train_loss_{name}.csv")
    G1_Val_loss = np.loadtxt(f"{path}/G1_Val_loss_{name}.csv")
    G2_Val_loss = np.loadtxt(f"{path}/G2_Val_loss_{name}.csv")
    G3_Val_loss = np.loadtxt(f"{path}/G3_Val_loss_{name}.csv")
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
    
    
    #G_total_train_loss = np.add(G1_train_loss, G2_train_loss)
    #G_total_val_loss = np.add(G1_Val_loss, G2_Val_loss)
    plt.subplot(2, 2, 4)
    plt.plot(range(1, n + 1), G3_train_loss, label="G3 Train")
    plt.plot(range(1, n + 1), G3_Val_loss, label="G3 Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.title("G3 Train vs Val Loss")

    plt.tight_layout()
    
    plot_filename = os.path.join("History_plots/", f"{name}_curves.png")
    plt.savefig(plot_filename)
    

