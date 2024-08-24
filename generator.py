import torch
import torch.nn as nn
from torchinfo import summary
import functools


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


'resUnet architecture'
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
        
        
        
        


        
        
        

#advanced generator arch 
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, self_attn_layer_indices=[]):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        #if 0 in self_attn_layer_indices:
        #    unet_block = nn.Sequential(unet_block, SelfAttn(ngf * 8, nn.ReLU(True), forward_outputs_attention=False))

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            #if i+1 in self_attn_layer_indices:
            #    unet_block = nn.Sequential(unet_block, SelfAttn(ngf * 8, nn.ReLU(True), forward_outputs_attention=False))
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        #if num_downs-4 in self_attn_layer_indices:
        #    unet_block = nn.Sequential(unet_block, SelfAttn(ngf * 8, nn.ReLU(True), forward_outputs_attention=False))
        
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        #if num_downs-3 in self_attn_layer_indices:
        #    unet_block = nn.Sequential(unet_block, SelfAttn(ngf * 4, nn.ReLU(True), forward_outputs_attention=False))

        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        #if num_downs-2 in self_attn_layer_indices:
        #    unet_block = nn.Sequential(unet_block, SelfAttn(ngf * 2, nn.ReLU(True), forward_outputs_attention=False))
                
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
        
        


#################################################################################

class Block(nn.Module):
    #down: True means in encoder part of generator, False means in decoder part
    #act: default activation function is relu
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride = 2, padding=1, bias=False, padding_mode="reflect")
            
            #down sampling after a single conv if encoding 
            if down
            #if in decoder part perform upsampling 
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride = 2, padding=1, bias=False),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2), 
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        #use dropout only in first 3 layers of UNet
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        
        
        #simple encoder-decoder architecture
        #encoder part, as layer become deeper, feature increase 
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False) #output img size 64
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=False) #32
        self.down3 = Block(features*4, features*8, down=True, act="leaky",use_dropout=False) #16
        self.down4 = Block(features*8, features*8, down=True, act="leaky",use_dropout=False) #8
        self.down5 = Block(features*8, features*8, down=True, act="leaky",use_dropout=False) #4
        self.down6 = Block(features*8, features*8, down=True, act="leaky",use_dropout=False) #2
        
        #bottleneck 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode="reflect"), #1x1 size
            nn.ReLU(),
        ) 
        
        
        #decoder part 
        #as layer coming out of bottleneck, feature size decrease
        self.up1 = Block(features*8, features*8, down=False,act="relu", use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False,act="relu", use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, act="relu",use_dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False, act="relu",use_dropout=False)
        self.up5 = Block(features*8*2, features*4, down=False, act="relu",use_dropout=False)
        self.up6 = Block(features*4*2, features*2, down=False, act="relu",use_dropout=False)
        self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=False)
        
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), #output pixel value between -1 to 1
        )
    
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        
        u1 = self.up1(bottleneck)
        #use skip connection to concatenate encoder output with decoder input
        #inherit d7 from encoder and apply to u1 section
        #look at UNET it totally make sense in terms of skip connection 
        #only difference is using single layer instead of double conv2 layer
        u2 = self.up2(torch.cat((u1,d7), dim=1))
        u3 = self.up3(torch.cat((u2,d6), dim=1))
        u4 = self.up4(torch.cat((u3,d5), dim=1))
        u5 = self.up5(torch.cat((u4,d4), dim=1))
        u6 = self.up6(torch.cat((u5,d3), dim=1))
        u7 = self.up7(torch.cat((u6,d2), dim=1))
        
        return self.final_up(torch.cat((u7,d1), dim=1))

#I maybe incorrect in the GAN architecture flowchart. Probably noise factor is not necessary but only the label image is required to provide (since we don't have multiclass but img itself is the label)
def test():
    x = torch.randn((1,3,256,256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
    model = Generator(in_channels=3, features=64)
    # Print size of each layer of the model
    #summary(model, input_size=(1, 3, 256, 256))
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, self_attn_layer_indices=[])
    summary(model, input_size=(1, 3, 256, 256))

