import torch 
import torch.nn as nn
from torchinfo import summary

'BN_ReLU block'
class BN_ReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    
    
class ResU_block(nn.Module):
    def __init__(self, in_channels, out_channels, s=1):
        super().__init__()
        "define the encoder block 2 and 3 (have same architecture)"
        self.bn_relu1 = BN_ReLU(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=s)
        self.bn_relu2 = BN_ReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) #stride for first conv is provided by user while second is 1
        
        #construct identity mapping 
        #final layer = conv2 + identity 
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=s)
    
    def forward(self, input):
        x = self.bn_relu1(input)
        x = self.conv1(x)
        x = self.bn_relu2(x)
        x = self.conv2(x)
        s = self.identity(input)
        #finish the addtion operation, also will be used to skip concatenate in future decoding layer
        skip = x + s
        return skip



class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #up sampling layer 
        'What is upsampling layer?'
        'It is a layer that increases the size of the input image through interpolation, completely mathematical operation without any param learning'
        'value in expoanded size is determiend by previous associated value'
        
        'Think as opposite of pooling layer!!!!! '
        'using "bilinear" algo (its a comoplex algo, but not too complex for computation power)for upsampling'
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #now normal decoder block, note each decoder block have same architecture as encoder2/3 block, except the cat and upsampling part
        #since cat, inc = inc+outc (512+256) (256+128....)
        self.decoder_block = ResU_block(in_channels+out_channels, out_channels)
        #default stride 1
        
        
        
    def forward(self, input, skipConnection):
        x = self.up(input)
        #get cat layers
        x = torch.cat([x, skipConnection], dim=1)
        x = self.decoder_block(x)
        return x



class ResUnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        "encoder block 1 construction"
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_relu1 = BN_ReLU(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #final layer = input layer identity mapping + addition to conv2
        #below is identity mapping hyperparam 
        self.input = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        
        "encoder block 2 and 3 (already construct)"
        self.encoder2 = ResU_block(64, 128, s=2)
        self.encoder3 = ResU_block(128, 256, s=2)
        
        'bridge connection / bottleneck (have exac same arch as encoder2 and 3)'
        self.encoder4 = ResU_block(256, 512, s=2)
        
        'decoder block 1 2 3 (since each decode block ahve same arch!!)'
        self.decoder1 = Decoder_block(512, 256)
        self.decoder2 = Decoder_block(256, 128)
        self.decoder3 = Decoder_block(128, 64)
        
        'output layers'
        #out channel is 3
        self.out = nn.Conv2d(64, 3, kernel_size=1, padding=0)
        #sigmoid is not mandatory, in unet we didn't use it so I won't use it in resUnet as well
        #self.sigmoid = nn.Sigmoid()
        
        
        
        
    def forward(self, input):
        'encoder block 1'
        x = self.conv1(input)
        x = self.bn_relu1(x)
        x = self.conv2(x)
        s = self.input(input) #NOT x!!!! because x have been changed!!
        #finish the addtion operation, also will be used to skip concatenate in future decoding layer
        skip1 = x + s
        
        'encoder block 2 and 3, recall that encoder block 2 and 3 have arch defined above'
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        #print(skip3.shape)
        
        'bridge connection/bottleneck'
        b = self.encoder4(skip3)
        
        'Decoder block 1'
        d1 = self.decoder1(b, skip3)
        d2 = self.decoder2(d1, skip2)
        d3 = self.decoder3(d2, skip1)
        
        'out layer'
        return self.out(d3)
        
        
        
         
        
        
        






if __name__ == "__main__":
    
    
    # Print size of each layer of the model
    #summary(model, input_size=(1, 3, 256, 256))
    #model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, self_attn_layer_indices=[])
    #summary(model, input_size=(1, 3, 256, 256))
    inputs = torch.randn(1, 3, 256, 256)
    model = ResUnet()
    #y = model(inputs)
    
    summary(model, input_size=(1, 3, 256, 256))
    
    #"C:\Users\gaoan\Desktop\res-Unet.py"
    
    
    