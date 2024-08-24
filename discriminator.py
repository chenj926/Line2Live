#discriminator is NOT conditioned!!! 
#It doesn't take input feature from generator!!! 

import torch
import torch.nn as nn
import torchinfo
from torchinfo import summary
import functools
import config
from collections import OrderedDict


class GaussianNoise(nn.Module):           # Try noise just for real or just for fake images.
    def __init__(self, std=0.3, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            #print("Not in training mode, noise not added")
            return x

class DiscriminatorWithNoise(nn.Module):
    def __init__(self, activation=nn.LeakyReLU, std=0.3, std_decay_rate=0):
        super().__init__()
        self.std = std
        self.std_decay_rate = std_decay_rate
        self.activation = activation
        self.stacks = nn.Sequential(*[
            self.downsample(32, bn=False),
            self.downsample(64),
            self.downsample(128),
            self.downsample(256),
            self.downsample(512),
            self.downsample(1024),
        ])

        self.head = nn.Sequential(OrderedDict([
            ('gauss', GaussianNoise(self.std, self.std_decay_rate)),
            ('linear', nn.LazyLinear(1)),
            # ('act', nn.Sigmoid()),        # removed for BCEWithLogitsLoss
        ]))

    def downsample(self, num_filters, bn=True, stride=2):
        layers = [
            GaussianNoise(self.std, self.std_decay_rate),
            nn.LazyConv2d(num_filters, kernel_size=4, stride=stride, bias=not bn, padding=1)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(num_filters))
        layers.append(self.activation(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.stacks(torch.cat([x,label], dim=1))
        x = x.flatten(1)
        x = self.head(x)
        return x
     



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, self_attn_layer_indices=[]):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            conv2d = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
            leakyReLU = nn.LeakyReLU(0.2, True)

            if type(norm_layer) != functools.partial and norm_layer.__name__ == 'SpectralNorm':
                sequence += [
                    norm_layer(conv2d),
                    leakyReLU
                ]
            else:
                sequence += [
                    conv2d,
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            #if n in self_attn_layer_indices:
            #    sequence.append(SelfAttn(ndf * nf_mult, nn.ReLU(True), forward_outputs_attention=False, where_to_log_attention=None))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        conv2d = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        leakyReLU = nn.LeakyReLU(0.2, True)
        final_conv2d = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw) # output 1 channel prediction map

        if type(norm_layer) != functools.partial and norm_layer.__name__ == 'SpectralNorm':
            sequence += [
                norm_layer(conv2d),
                leakyReLU,
                norm_layer(final_conv2d)
            ]
        else:
            sequence += [
                conv2d,
                norm_layer(ndf * nf_mult),
                leakyReLU,
                final_conv2d
            ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x, label):
        """Standard forward."""
        x = torch.cat([x,label], dim=1)
        return self.model(x)
    
    
    
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, use_noise=True, std=0.3, decay_rate=0):
        super().__init__()
        self.conv = nn.Sequential(
            #"reflect": reflect the input by padding with the input mirrored along the edges
            GaussianNoise(std, decay_rate) if use_noise else nn.Identity(),
            
            
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            
            #add dropout to prevent discriminator being too powerful
            nn.Dropout(0.2) if config.USE_DROPOUT else nn.Identity()
            
            
            
        )
        
    def forward(self,x):
        return self.conv(x)



class Discriminator(nn.Module):
    #send in 256 inputs and get size 30 output after 4 conv block 
    #(calculated by Conv2d equation with feature size)
    def __init__(self, in_channels=3,features=[64,128,256,512], use_noise=True, std=0.3, decay_rate=0):
        super().__init__()
        #initial block is different than conv! a special start
        self.initial = nn.Sequential(
            GaussianNoise(std, decay_rate) if use_noise else nn.Identity(),
            
            #why in_channels*2? because we are going to concatenate the input image (the "condition") with the output of the generator! We are actually sending 2 imgs here
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            
            #try to add some dropout to prevent discriminator being too powerful 
            nn.Dropout(0.2) if config.USE_DROPOUT else nn.Identity()
        )
        
        #now construct rest 4 conv2 block
        layers=[]
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2, use_noise=True), #do stride of 1 for last layer
            )
            in_channels = feature
            
        #ensure the output is a single value per patch
        #just add one additional conv2d 
        layers.append(
            GaussianNoise(std, decay_rate) if use_noise else nn.Identity(),
            
            
        )
        layers.append(
            #output a single channel 
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
        )
            
        #unpack layers block to model 
        self.model = nn.Sequential(*layers)
        
    
    #send in sketch (as label) + real gray img/colorful img/img generated by generator
    #to make predictions 
    def forward(self, x, label):
        #concatenate the input image with the output of the generator
        x = torch.cat([x,label], dim=1)
        x = self.initial(x)
        #return a probability of real or fake
        return self.model(x)






    
    

#test if descriminator work 
def test():
    x = torch.randn((1,3,256,256)) #[B, C, H, W]
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    preds = model(x,y)
    #output shape should be [1,1,X,X] 
    #for each patch we have to output a single vaule of 0-1 (X is number of patch)
    print("Final output shape", preds.shape)
    
if __name__ == "__main__":
    test()
    
    model = Discriminator()
    # Print size of each layer of the model
    summary(model, input_size=((1, 3, 256, 256), (1, 3, 256, 256)))
    
    model = NLayerDiscriminator(input_nc = 3+3, ndf = 64, norm_layer=nn.BatchNorm2d)
    summary(model, input_size=((1, 3, 256, 256), (1, 3, 256, 256)))
        
        