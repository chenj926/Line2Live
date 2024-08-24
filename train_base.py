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
from generator import Generator, UnetGenerator, ResUnet
from discriminator import Discriminator, NLayerDiscriminator, DiscriminatorWithNoise




def train_loop(disc, gen, loader, optimizer_disc, optimizer_gen, l1, bce, g_scaler,d_scaler):
    
    
    #batch size, C, H, W
    gen.train()
    disc.train()
    
    
    #(x,y) is a batch of sketch (x) and target img (y)
    #idx represent current batch #
    for idx, (x,y) in enumerate(loader):
        
        
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        
        #print(noise.size(), y.size())
        

        #print("Batch: ", idx)
        
        #train discriminator
        #cast operation to mixed precision float16
        with torch.cuda.amp.autocast(dtype=torch.float16):
            if torch.isnan(x).any():
                print("NaN detected in input data x")
                continue
            
            y_fake = gen(x)
            #print(max(y_fake[0][0].flatten()), min(y_fake[0][0].flatten()))
            
            
            
            if torch.isnan(y_fake).any():
                print("NaN detected in y_fake generation")
                continue
            
            
            
            #get the probability of real img and fake img
            D_real_logits = disc(x,y)
            D_fake_logits = disc(x,y_fake.detach()) #detach is necessary 
            
            if config.STABILITY:
                D_real_logits = torch.clamp(D_real_logits, min=1e-7, max=1-1e-7)
                D_fake_logits = torch.clamp(D_fake_logits, min=1e-7, max=1-1e-7)
            
            if torch.isnan(D_fake_logits).any():
                print("NaN detected in D_fake_logits in disc")
                continue
            if torch.isnan(D_real_logits).any():
                print("NaN detected in D_real_logits in disc")
                continue
            
            #evalulate loss, since pass in all real imgs, the target is all 1 (use torch.ones_like to create tensor of 1s with same shape as D_real_logits)
            D_real_loss = bce(D_real_logits, torch.ones_like(D_real_logits)*0.9) #introduce label smoothing, less confident about real images
            if torch.isnan(D_real_loss).any():
                print("NaN detected in D_real_loss in disc")
                continue
            
            #same for fake loss, except target all 0
            D_fake_loss = bce(D_fake_logits, torch.zeros_like(D_fake_logits)) #can also use torch.ones_like(D_fake_logits)*0.1 to encourage discrm be less confident about fake images, less confident about fake images 
            if torch.isnan(D_fake_loss).any():
                print("NaN detected in D_fake_loss in disc")
                continue
            
            
            
            #sometimes ppl divide the loss by 2....................
            D_loss = (D_real_loss + D_fake_loss)
        
        disc.zero_grad()
        #steps to perform float16 training
        d_scaler.scale(D_loss).backward()
        
        if config.STABILITY:
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        
        d_scaler.step(optimizer_disc)
        d_scaler.update()
        
        
        
        #train generator
        with torch.cuda.amp.autocast(dtype=torch.float16):
            #trying to fool discriminator, therefore the fake_logit computed from disc should have target all of 1 !!!!
            #Note while in disc training, want to align fake_logit with 0!!!! Battle... 
            
            D_fake_logits = disc(x,y_fake)
            
            if config.STABILITY:
                D_fake_logits = torch.clamp(D_fake_logits, min=1e-7, max=1-1e-7)
            if torch.isnan(D_fake_logits).any():
                print("NaN detected in D_fake_logits in GEN")
                continue
            
            G_fake_loss = bce(D_fake_logits, torch.ones_like(D_fake_logits))
            if torch.isnan(G_fake_loss).any():
                print("NaN detected in G_fake_loss in GEN")
                continue
           
                
            #sum of diff in y_pred and y
            L1_loss = l1(y_fake, y) * config.L1_LAMBDA
            #total loss
            G_loss = G_fake_loss + L1_loss
        
        
        optimizer_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        
        if config.STABILITY:
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        
        
        g_scaler.step(optimizer_gen)
        g_scaler.update()
        
        
    print("TRAIN \n G_loss: ", G_loss.item(), "\n D_loss: ", D_loss.item())
    return G_loss.item(), D_loss.item()
        
    
    

def validate_loop(d1, g1, loader, l1, bce, epoch):
    g1.eval()
    d1.eval()
    g1_loss = 0
    d_loss = 0
    
    
        
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                
            y_fake = g1(x)
            
                
            D_real_logits = d1(x, y)
            D_fake_logits_color = d1(x, y_fake.detach())
            
                
            D_real_loss = bce(D_real_logits, torch.ones_like(D_real_logits))
            D_fake_loss_color = bce(D_fake_logits_color, torch.zeros_like(D_fake_logits_color))
           
                
            D_loss = D_real_loss + D_fake_loss_color 
                
            G_fake_loss_color = bce(D_fake_logits_color, torch.ones_like(D_fake_logits_color))
            L1_color = l1(y_fake, y) * config.L1_LAMBDA
            G2_loss = G_fake_loss_color + L1_color
                
            
            g1_loss += G2_loss.item()
            d_loss += D_loss.item()
        
        g1_loss /= len(loader)
        d_loss /= len(loader)
        
        print("VALIDATION \n G_loss: ", g1_loss, "\n D_loss: ", d_loss)
        
        return g1_loss, d_loss
     
    
    
    
def main():
    disc = Discriminator(in_channels=3, use_noise=True, std=0.3, decay_rate=0).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    #disc = NLayerDiscriminator(input_nc = 3+3, ndf = 64, norm_layer=nn.BatchNorm2d).to(config.DEVICE)
    #disc = DiscriminatorWithNoise(std=0.5, std_decay_rate=0).to(config.DEVICE)
    #gen = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, self_attn_layer_indices=[]).to(config.DEVICE)
    #gen = ResUnet().to(config.DEVICE)
    #gen2 = Generator(in_channels=3).to(config.DEVICE)
    
    
    
    #can configure different learning rate for gen and disc
    optimizer_disc = optim.Adam(disc.parameters(), lr=0.0002, betas=config.BETAS, weight_decay=0) #note betas is a play with momentum can change here 
    optimizer_gen = optim.Adam(gen.parameters(), lr=0.0002, betas=config.BETAS, weight_decay=0)
    #optimizer_gen2 = optim.Adam(gen2.parameters(), lr=0.0002, betas=config.BETAS)
    
    
    #standard GAN loss
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    #GPloss didn't work well with patchGan
    
    
    
    #load the model for hyperparam tuning
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, optimizer_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, optimizer_disc, config.LEARNING_RATE)
        #load_checkpoint(config.CHECKPOINT_GEN2, gen2, optimizer_gen2, config.LEARNING_RATE)
        
    
    #change into train!!!!!!
    if not config.CUFS:
        test_dataset = MapDataset(sketch_dir='Student_Only_Aug/MARK!!_sketch_test_student_DEMO', target_dir='Student_Only_Aug/MARK!!_photos_test_student_DEMO')
    else: 
        test_dataset = MapDataset(sketch_dir='CUFS_Only/test_sketch_removeShadow', target_dir='CUFS_Only/test_photo_color')
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Test dataset loaded")
    
    if config.TEST_ONLY and config.LOAD_MODEL:
        
        name = "Student_Noise_Base_DataAug"
        if not os.path.exists(f"Final_Generation/g_{name}"):
            os.makedirs(f"Final_Generation/g_{name}")
        final_save_all_ColorOnly(gen, test_loader, folderName=f"Final_Generation/g_{name}")
        print("All test files saved successfully")
        exit()
    
    
    
    #LOAD DATASET and Save transformed images 
    if not config.CUFS and not config.DataAug:
        train_dataset = MapDataset(sketch_dir='Student_Only/sketch_train_studentSaved', target_dir='Student_Only/photos_train_student_whiteBG')
    elif not config.CUFS and config.DataAug and config.Saturation:
        train_dataset = MapDataset(sketch_dir='Student_Only_Aug/sketch_train_studentSaved', target_dir='Student_Only_Aug/photos_train_student_whiteBG')
    elif not config.CUFS and config.DataAug and not config.Saturation:
        train_dataset = MapDataset(sketch_dir='Student_Only_Aug/sketch_train_studentSaved_NoSat', target_dir='Student_Only_Aug/photos_train_student_whiteBG_NoSat')
    else: 
        train_dataset = MapDataset(sketch_dir='CUFS_Only/train_sketch_removeShadow', target_dir='CUFS_Only/train_photo_color')
    
    #construct dataloader
    #batch size = 32 if not ResUnet, else 24
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=config.NUM_WORKERS)
    print("Train dataset loaded")
    
    
    
    
    
    
    
    #exit()
    
    #verify val dataset
    #modify functino to save intermediate generated img 
    
    #load validation dataset
    if not config.CUFS:
        val_dataset = MapDataset(sketch_dir='Student_Only_Aug/sketch_val_student_DEMO', target_dir='Student_Only_Aug/photos_val_student_DEMO')
    else:
        val_dataset = MapDataset(sketch_dir='CUFS_Only/val_sketch_removeShadow', target_dir='CUFS_Only/val_photo_color')
    ############################################################################
    #save_transformed_images(val_dataset, save_sketch_dir='sketch_val_studentSaved', save_tar_dir='photos_val_studentSaved')
    ############################################################################
    
    
    
   
    
    
    
    
    
    #only validate one img at a time
    if not config.CUFS:
        val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False)
    else:
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

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
        G1_loss,  D_loss = train_loop(disc, gen,   train_loader, optimizer_disc, optimizer_gen,  L1_LOSS, BCE, g1_scaler, d_scaler)
        G1_val_loss,  D_val_loss = validate_loop(disc, gen,  val_loader, L1_LOSS, BCE, epoch)
        
        
        
        
        
        
        
        
        G1_train_loss_all[epoch] = G1_loss
        #G2_train_loss_all[epoch] = G2_loss
        D_train_loss_all[epoch] = D_loss
        G1_Val_loss_all[epoch] = G1_val_loss
        #G2_Val_loss_all[epoch] = G2_val_loss
        D_Val_loss_all[epoch] = D_val_loss
        
        
        #save model checkpoint every # epoch
        #and save the last model config 
        if (config.SAVE_MODEL and epoch == config.NUM_EPOCHS - 1) or (epoch % 50 == 0 and epoch != 0  and config.SAVE_MODEL):
            save_checkpoint(gen, optimizer_gen, filename=f"Generators/g1_{name}_epoch_{epoch}.pth.tar")
            save_checkpoint(disc, optimizer_disc, filename=f"Discriminators/d_{name}_epoch_{epoch}.pth.tar")
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
    
    