#from lib
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader





from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time 
import os


#from files
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples, final_save_all_ColorOnly, plot_training_curve_base
from dataset import MapDataset, save_transformed_images, save_single_transformed
from generator import Generator
from discriminator import Discriminator




def train_loop(critic, gen, loader, optimizer_critic, optimizer_gen, l1, g_scaler,d_scaler):
    
    
    #(x,y) is a batch of sketch (x) and target img (y)
    #idx represent current batch #
    for idx, (x,y) in enumerate(loader):
            
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    
        
        #print(x.type(), y.type())
        
        #print("Batch: ", idx)
        
        #train discriminator
        #cast operation to mixed precision float16
        #with torch.cuda.amp.autocast(dtype=torch.float16):
            
        for _ in range(config.CRITIC_ITERATIONS):
            y_fake = gen(x)
            critic_real = critic(x, y).reshape(-1)
            critic_fake = critic(x, y_fake.detach()).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) #maximize loss critic
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()
            
            for p in critic.parameters():
                p.data.clamp_(-config.WEIGHT_CLIP, config.WEIGHT_CLIP)
            
          
        
        
        output = critic(x, y_fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()
        
        
        
    print("TRAIN \n G_loss: ", loss_gen.item(), "\n D_loss: ", loss_critic.item())
    return loss_gen.item(), loss_critic.item()
        
    
    

def validate_loop(critic, g1, loader, l1, epoch):
    g1.eval()
    critic.eval()
    loss_gen = 0
    loss_critic = 0
    
    
    
    for idx, (x,y) in enumerate(loader):
            
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    
        
        #print(x.type(), y.type())
        
        #print("Batch: ", idx)
        
        #train discriminator
        #cast operation to mixed precision float16
        #with torch.cuda.amp.autocast(dtype=torch.float16):
            
        for _ in range(config.CRITIC_ITERATIONS):
            y_fake = g1(x)
            critic_real = critic(x, y).reshape(-1)
            critic_fake = critic(x, y_fake.detach()).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) #maximize loss critic
            
         
            
          
        
        
        output = critic(x, y_fake).reshape(-1)
        loss_gen = -torch.mean(output)
        
        print("VALIDATION \n G_loss: ", loss_gen.item(), "\n D_loss: ", loss_critic.item())
        
        return loss_gen, loss_critic
     
    

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
    
def main():
    critic = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    #gen2 = Generator(in_channels=3).to(config.DEVICE)
    
    
    
    #can configure different learning rate for gen and disc
    optimizer_critic = optim.RMSprop(critic.parameters(), lr=5e-5) #note betas is a play with momentum can chang here 
    optimizer_gen = optim.RMSprop(gen.parameters(), lr=5e-5)
    #optimizer_gen2 = optim.Adam(gen2.parameters(), lr=0.0002, betas=config.BETAS)
    
    
    initialize_weights(critic)
    initialize_weights(gen)
    
    #standard GAN loss
    #BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    #GPloss didn't work well with patchGan
    
    
    
    #load the model for hyperparam tuning
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, optimizer_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, critic, optimizer_critic, config.LEARNING_RATE)
        #load_checkpoint(config.CHECKPOINT_GEN2, gen2, optimizer_gen2, config.LEARNING_RATE)
        
    
    test_dataset = MapDataset(sketch_dir='CUFS_Only/test_sketch_removeShadow', target_dir='CUFS_Only/test_photo_color')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Test dataset loaded")
    
    if config.TEST_ONLY and config.LOAD_MODEL:
        
        name = "Multi_FirstTrial"
        if not os.path.exists(f"Final_Generation/g_{name}"):
            os.makedirs(f"Final_Generation/g_{name}")
        final_save_all_ColorOnly(gen, test_loader, folderName=f"Final_Generation/g_{name}")
        
        exit()
    
    
    
    #LOAD DATASET and Save transformed images 
    train_dataset = MapDataset(sketch_dir='CUFS_Only/train_sketch_removeShadow', target_dir='CUFS_Only/train_photo_color')
    ############################################################################
    #Note: may only need to run once if train_dataset didn't change
    #after applied resize and normalize, save the transformed images
    #THIS STEP ONLY FOR BACKGROUND REMOVAL!!! AND GrayScale conversion! 
    ############################################################################
    #save_transformed_images(train_dataset, save_sketch_dir='sketch_train_studentSaved', save_tar_dir='photos_train_studentSaved')
    
    
    
    #construct dataloader
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=config.NUM_WORKERS)
    
    print("Train dataset loaded")
    
    
    
    
    
    
    
    #exit()
    
    #verify val dataset
    #modify functino to save intermediate generated img 
    
    #load validation dataset
    val_dataset = MapDataset(sketch_dir='CUFS_Only/val_sketch_removeShadow', target_dir='CUFS_Only/val_photo_color')
    ############################################################################
    #save_transformed_images(val_dataset, save_sketch_dir='sketch_val_studentSaved', save_tar_dir='photos_val_studentSaved')
    ############################################################################
    
    
    
   
    
    
    
    
    
    #only validate one img at a time
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    print("Val dataset loaded")
    
    '''
     # Visualize a batch of dataset pairs in train_loader
    sample_batch = next(iter(train_loader))
    x, y = sample_batch[0], sample_batch[1]
    #print(x.shape, y.shape, y_gray.shape)
    #print(x.dtype, y.dtype, y_gray.dtype)
    
    
    #set batchsize to corresponded batch size of val/train loader
    fig, axes = plt.subplots(8, 3, figsize=(10, 10))
    for i in range(8):
        axes[i, 0].imshow(x[i].permute(1, 2, 0))
        axes[i, 0].set_title('Sketch')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(y[i].permute(1, 2, 0))
        axes[i, 1].set_title('Target Image')
        axes[i, 1].axis('off')
        #axes[i, 2].imshow(y_gray[i].permute(1, 2, 0))
        #axes[i, 2].set_title('Gray Image')
        #axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()
    
    '''
    
    
    
    #perform float16 training
    g1_scaler = torch.cuda.amp.GradScaler()
    #g2_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    #save all loss for plotting
    G1_train_loss_all = np.zeros(config.NUM_EPOCHS)
    #G2_train_loss_all = np.zeros(config.NUM_EPOCHS)
    D_train_loss_all = np.zeros(config.NUM_EPOCHS)
    G1_Val_loss_all = np.zeros(config.NUM_EPOCHS)
    #G2_Val_loss_all = np.zeros(config.NUM_EPOCHS)
    D_Val_loss_all = np.zeros(config.NUM_EPOCHS)
    
    
    start_time = time.time()
    
    #saving name
    #####################################
    name = config.NAME
    #####################################
    
    torch.manual_seed(1000)
    torch.autograd.set_detect_anomaly(True)

    #train the model
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
        
        print("\n Epoch ", epoch)
        G1_loss,  D_loss = train_loop(critic, gen,   train_loader, optimizer_critic, optimizer_gen,  L1_LOSS,  g1_scaler, d_scaler)
        G1_val_loss,  D_val_loss = validate_loop(critic, gen,  val_loader, L1_LOSS, epoch)
        
        
        
        
        
        
        
        
        G1_train_loss_all[epoch] = G1_loss
        #G2_train_loss_all[epoch] = G2_loss
        D_train_loss_all[epoch] = D_loss
        G1_Val_loss_all[epoch] = G1_val_loss
        #G2_Val_loss_all[epoch] = G2_val_loss
        D_Val_loss_all[epoch] = D_val_loss
        
        
        #save model checkpoint every # epoch
        #and save the last model config 
        if (config.SAVE_MODEL and epoch == config.NUM_EPOCHS - 1) or (config.SAVE_MODEL and epoch % 50 == 0 and epoch != 0):
            save_checkpoint(gen, optimizer_gen, filename=f"Generators/g1_{name}_epoch_{epoch}.pth.tar")
            save_checkpoint(critic, optimizer_critic, filename=f"Discriminators/d_{name}_epoch_{epoch}.pth.tar")
            #save_checkpoint(gen2, optimizer_gen2, filename=f"Generators/g2_{name}_epoch_{epoch}.pth.tar")
            
        
        #save some validation generated examples
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS - 1 or epoch == 0:
            
            # Create directory if it doesn't exist
            if not os.path.exists(f"validation_gen_examples"):
                os.makedirs(f"validation_gen_examples")
            folder = f"validation_gen_examples/{name}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            #create train dir 
            if not os.path.exists(f"train_gen_examples"):
                os.makedirs(f"train_gen_examples")
            folder2 = f"train_gen_examples/{name}"
            if not os.path.exists(folder2):
                os.makedirs(folder2)
                

            save_some_examples(gen, val_loader, epoch, folder)
            save_some_examples(gen, train_loader, epoch, folder2)
            
            
        #save all training examples when model finished training 
        if epoch == config.NUM_EPOCHS - 1:
            if not os.path.exists(f"Final_Generation/g_{name}"):
                os.makedirs(f"Final_Generation/g_{name}")
            final_save_all_ColorOnly(gen, test_loader, folderName=f"Final_Generation/g_{name}")
    
    
    end_time = time.time()
    print(f"Time taken to Train: {end_time - start_time}")
    
    np.savetxt(f"History/G1_train_loss_{name}.csv", G1_train_loss_all)
    #np.savetxt(f"History/G2_train_loss_{name}.csv", G2_train_loss_all)
    np.savetxt(f"History/D_train_loss_{name}.csv", D_train_loss_all)
    np.savetxt(f"History/G1_Val_loss_{name}.csv", G1_Val_loss_all)
    #np.savetxt(f"History/G2_Val_loss_{name}.csv", G2_Val_loss_all)
    np.savetxt(f"History/D_Val_loss_{name}.csv", D_Val_loss_all)
    
    
    
    plot_training_curve_base("History")
    print("All History plot saved")
    
    
if __name__ == "__main__":
    main()
    
    