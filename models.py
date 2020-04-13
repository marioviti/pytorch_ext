import torch.nn as nn
import numpy as np
from .conv import Conv, UpConv, MaxPooling, ConvBlock

class Classifier(nn.Module):
    def __init__(self, input_channels, feature_channels, output_channels, padding='valid',
                 activation=nn.ReLU(inplace=True), mode='3d', batch_norm=True):
        super(Classifier, self).__init__()
        self.feature_pool = nn.AdaptiveMaxPool3d([1,1,1]) if mode=='3d' else nn.AdaptiveMaxPool2d([1,1])
        self.bn  = nn.BatchNorm3d(input_channels) if mode=='3d' else nn.BatchNorm2d(input_channels)
        self.conv0 = Conv(input_channels, feature_channels, kernel_size=1, mode=mode, padding=padding)
        self.conv1 = Conv(feature_channels, max(feature_channels//2,1), kernel_size=1, mode=mode, padding=padding)
        self.conv2 = Conv(max(feature_channels//2,1), max(feature_channels//4,1), kernel_size=1, mode=mode, padding=padding)
        self.conv3 = Conv(max(feature_channels//4,1), output_channels, kernel_size=1, mode=mode, padding=padding)
        self.activation = activation
        self.attention_activation = nn.Sigmoid() if output_channels == 1 else nn.Softmax()
        
    def forward(self,x):
        bs = x.shape[0]
        x = self.bn(self.feature_pool(x))
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x).view(bs,-1)
        return x
    
    def attention_maps(self,x):
        x = self.bn(x)
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        x = self.attention_activation(x)
        return x

class ConvEncoder(nn.Module):
    
    def __init__(self, input_channels, stages=4, conv_per_stage=3, 
                 feature_channels=16, feature_expansion_rate=2,
                 residual=True, batch_norm=True, padding='same',
                 activation=nn.ReLU(inplace=True), mode='2d',
                 ):
        """
        Multi Stage Convolutional Encoder
        Each hidden stage consists of a convolution + activation + maxpooling
        an output stage (context or code) consists of a convolution + activation.
        #{stages} = #{hidden stages} + #{code stage}.
        """
        super(ConvEncoder, self).__init__()
        in_c, out_c = input_channels, feature_channels
        convs_list = [ Conv(in_c, out_c, mode=mode) ]
        self.hidden_channels = [out_c]
        for i in range(1,stages):
            in_c, out_c = out_c, np.round(out_c*feature_expansion_rate).astype(np.int)
            convs_list += [ ConvBlock(conv_per_stage, in_c, out_c,
                                      padding=padding,
                                      batch_norm=batch_norm, residual=residual,
                                      conv_mode=mode, activation=activation) ]
            self.hidden_channels += [out_c]
        self.input_channels = input_channels
        self.output_channels = out_c
        self.conv_blocks = nn.ModuleList(convs_list)
        self.max_pool = MaxPooling(mode=mode)

    def forward(self, x):
        out = x
        self.hidden_features = []
        for conv in self.conv_blocks[:-1]:
            out = conv(out)
            self.hidden_features += [out]
            out = self.max_pool(out)
        out = self.conv_blocks[-1](out)
        return out
    
class ConvDecoder(nn.Module):
    def __init__(self, output_channels, stages=4, conv_per_stage=3, 
                 feature_channels=16, feature_expansion_rate=2, mode='2d',
                 padding='same', batch_norm=True, residual=True,
                 activation=nn.ReLU(inplace=True), skips=False):
        """
        Multi Stage Convolutional Decoder
        Each hidden stage consists of a deconvoltion + activation
        an output stage consists of a convolution.
        #{stages} = #{hidden stages} + #{output stage}.
        """
        super(ConvDecoder, self).__init__()
        upconv_list, conv_block_list, self.hidden_channels = [], [], []
        in_c, out_c = feature_channels, np.round(feature_channels*feature_expansion_rate).astype(np.int)
        for i in range(0,stages-1):
            upconv_list += [ UpConv(in_c, out_c, mode=mode) ]
            conv_block_list += [ ConvBlock(conv_per_stage, out_c, out_c,
                                           padding=padding, batch_norm=batch_norm, residual=residual,
                                           conv_mode=mode, activation=activation) ]
            self.hidden_channels += [ out_c ]
            in_c, out_c = out_c, np.round(out_c*feature_expansion_rate).astype(np.int)
        self.input_channels = feature_channels
        self.output_channels = output_channels
        self.output_conv = Conv(in_c, output_channels, mode=mode) 
        self.up_convs = nn.ModuleList(upconv_list)
        self.conv_blocks = nn.ModuleList(conv_block_list)
        self.act = activation
        self.skips = skips
    
    def forward(self, x):
        self.hidden_features = []
        if self.skips:
            # x is a list
            input_ = x[0]
            skips = x[1:]
            out = input_
            for skip_i, up_conv, conv_bock in zip(skips, self.up_convs, self.conv_blocks):
                out = self.act(up_conv(out))
                out = conv_bock( out + skip_i )
                self.hidden_features += [out]
            out = self.output_conv(out)
            return out
        else:
            out = x
            for up_conv, conv_bock in zip(self.up_convs, self.conv_blocks):
                out = self.act(up_conv(out))
                out = conv_block(out)
                self.hidden_features += [out]
            out = self.output_conv(out)
            return out
        
class ConvAutoEncoder(nn.Module):
    def __init__(self, input_channels, stages=4, feature_channels=16, feature_expansion_rate=2, mode='2d',
                 activation=nn.ReLU(inplace=True)):
        super(ConvAutoEncoder, self).__init__()
        self.conv_encoder = ConvEncoder(input_channels, stages=stages, feature_channels=feature_channels,
                                        feature_expansion_rate=feature_expansion_rate, mode=mode,
                                        activation=activation)
        feature_channels_decoder = self.conv_encoder.hidden_channels[-1]
        self.conv_decoder = ConvDecoder(input_channels, stages=stages, feature_channels=feature_channels_decoder, 
                                        feature_expansion_rate=np.float(1/feature_expansion_rate), 
                                        mode=mode, activation=activation)
        
    def forward(self, x):
        self.hidden_features = [self.conv_encoder(x)]
        out = self.conv_decoder(self.hidden_features[0])
        return out
    
class ConvUnet(nn.Module):
    def __init__(self, input_channels, output_channels, stages=4, feature_channels=16, feature_expansion_rate=2, mode='2d',
                 activation=nn.ReLU(inplace=True)):
        super(ConvUnet, self).__init__()
        self.conv_encoder = ConvEncoder(input_channels, stages=stages, feature_channels=feature_channels,
                                        feature_expansion_rate=feature_expansion_rate, mode=mode,
                                        activation=activation)
        feature_channels_decoder = self.conv_encoder.hidden_channels[-1]
        self.conv_decoder = ConvDecoder(output_channels, stages=stages, feature_channels=feature_channels_decoder, 
                                        feature_expansion_rate=np.float(1/feature_expansion_rate), skips=True,
                                        mode=mode, activation=activation)
        self.activation = activation
        
    def set_skips(self, val):
        self.conv_decoder.skips = val
        
    def forward(self, x):
        self.hidden_features = [self.activation(self.conv_encoder(x))]
        if self.conv_decoder.skips:
            out = self.hidden_features + self.conv_encoder.hidden_features[::-1]
            out = self.conv_decoder(out)
        else:
            out = self.conv_decoder(self.hidden_features[0])
            return out
        return out