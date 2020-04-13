import torch as th
import torch.nn as nn    
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
import numpy.matlib as mtlb
from .conv_utils import *

class NNWl(nn.Module):
    """Implementation of https://arxiv.org/pdf/1812.00572.pdf U is set to 1"""
    def __init__(self, input_channels, output_channels, wl=0.5, ww=1.0, eps=1e-3):
        super(NNWl, self).__init__()
        self._op0 = nn.Conv3d(input_channels, output_channels, kernel_size=1, groups=input_channels)
        self._op0.weight.data.fill_(1)
        self._op0.weight.data.mul_( (2./ww) * np.log(1./eps - 1.) ) # log_e(x)
        self._op0.bias.data.fill_(1)
        self._op0.bias.data.mul_( ((-2.*wl)/ww) * np.log(1./eps - 1.) ) # log_e(x)
        self.act = th.nn.Sigmoid()
        
    def forward(self,x):
        return self.act(self._op0(x))

class EmbeddingCat(nn.Module):
    def __init__(self):
        super(EmbeddingCat, self).__init__()
        
    def forward(self, emb_x):
        emb,x = emb_x
        batch_size_x, feature_x = x.shape[:2]
        spatial_dims_x = x.shape[2:]
        batch_size_emb, feature_emb = emb.shape[:2]
        len_shape_x = len(x.shape)
        len_shape_emb = len(emb.shape)
        for i in range(len_shape_x-len_shape_emb): emb = emb[...,np.newaxis]
        assert batch_size_emb == batch_size_x
        emb = emb.expand([ batch_size_emb, feature_emb, *spatial_dims_x ])
        return th.cat([emb, x], dim=1)

class NNHist(nn.Module):
    def __init__(self, K, B, dims=3, reduce=True, keep_dims=True, min_xk=-1, max_xk=1):
        """Implementation of learnable histograms https://arxiv.org/abs/1804.09398.
        args:
            K: number of feature channel in input.
            B: number of bins for the histogram.
            reduce: reduce features via average (normalized histogram)
        forward:
            outputs feature bins are NOT reduced (average, sum).
        """
        super(NNHist, self).__init__()
        conv_fun = [ nn.Conv1d, nn.Conv2d, nn.Conv3d ][dims-1]
        self.mu = conv_fun(K,B*K, kernel_size=1, groups=K, bias=True)
        bin_width = (max_xk - min_xk)/(B-1)
        mu_data = min_xk + th.range(0,B-1)*bin_width 
        self.mu.bias.data = - mu_data.repeat(K)
        self.mu.weight.data.fill_(1)
        self.mu.weight.requires_grad = False
        self.w = conv_fun(B*K,B*K, kernel_size=1, groups=B*K, bias=False)
        self.w.weight.data.fill_(1)
        self.w.weight.data.mul_(1/float(bin_width))
        self.act = nn.ReLU(inplace=True)
        self.reduce = reduce
        self.dims = dims
        self.keep_dims = keep_dims
        
    def forward(self,x):
        a = self.act(1 - self.w(th.abs(self.mu(x))))
        if self.reduce: 
            dim_list = list(range(-self.dims,0))
            a = a.mean(dim_list, self.keep_dims)
        return a
    
    def clamp_after_update(self):
        """
        can be called after update to avoid negative bins
        """
        self.w.data.clamp_(1e-8)

# Convolution non trainable layers

kernels3D = {
               'dx' : lambda : np.array([[[-1,0,1]]]).transpose(2,0,1),
               'dy' : lambda : np.array([[[-1,0,1]]]).transpose(0,2,1),
               'dz' : lambda : np.array([[[-1,0,1]]]),
               'Gx' : lambda sigma: norm.pdf(np.linspace(-2*sigma,2*sigma,sigma*2+sigma%2),0,sigma)[..., np.newaxis, np.newaxis],
               'Gy' : lambda sigma: norm.pdf(np.linspace(-2*sigma,2*sigma,sigma*2+sigma%2),0,sigma)[np.newaxis, :, np.newaxis],
               'Gz' : lambda sigma: norm.pdf(np.linspace(-2*sigma,2*sigma,sigma*2+sigma%2),0,sigma)[np.newaxis, np.newaxis],
            }

conv_funs = { 
                1: F.conv1d,
                2: F.conv2d, 
                3: F.conv3d 
            }

class Pyramid(nn.Module):
    
    def __init__(self, input_channels, levels=1, dimensions=3):
        super(Pyramid, self).__init__()
        kernel_blur = np.ones([3]*dimensions)
        self.dimensions = dimensions
        self.conv_blur = Convolve(kernel_blur, input_channels)
        self.levels = levels
        
    def forward(self,x):
        out = []
        for level in range(self.levels):
            x_ = self.conv_blur(x)
            dog = x_ - ((3**self.dimensions-1)*x)
            if self.dimensions == 3:
                x = x_[:,:,::2,::2,::2]
            if self.dimensions == 2:
                x = x_[:,:,::2,::2]
            out += [[x_,dog]]
        return out

class Convolve(nn.Module):
    
    def __init__(self, kernel, input_channels, stride=1, padding=1, dilation=1):
        super(Convolve, self).__init__()
        self.convfun = conv_funs[len(kernel.shape)]
        kernel = kernel[np.newaxis,np.newaxis]
        kernel = np.concatenate([kernel]*input_channels, axis=0)
        # THIS IS SO IMPORTANT ....
        self.kernel_tensor = th.nn.Parameter(th.from_numpy(kernel).float())
        self.kernel_tensor.requires_grad = False
        self.input_channels = input_channels
        self.dilation, self.stride, self.padding = dilation, stride, padding
        
    def forward(self, x):
        x = self.convfun(x, self.kernel_tensor, groups=self.input_channels, 
                         stride=self.stride, padding=self.padding, dilation=self.dilation)
        return x
    
class GaussianBlur3D(nn.Module):
    
    def __init__(self, sigma, input_channels):
        super(GaussianBlur3D, self).__init__()
        if sigma%2 == 0:
            print('warning: not symmetric, fallback to sigma - 1')
            sigma -= 1
        Gx = kernels3D['Gx'](sigma)
        Gy = kernels3D['Gy'](sigma)
        Gz = kernels3D['Gz'](sigma)
        self.padx = nn.ReplicationPad3d(list(reversed([sigma,sigma,0,0,0,0])))
        self.pady = nn.ReplicationPad3d(list(reversed([0,0,sigma,sigma,0,0])))
        self.padz = nn.ReplicationPad3d(list(reversed([0,0,0,0,sigma,sigma])))
        self.conv_Gx = Convolve(Gx/Gx.sum(), input_channels, padding=0)
        self.conv_Gy = Convolve(Gy/Gy.sum(), input_channels, padding=0)
        self.conv_Gz = Convolve(Gz/Gz.sum(), input_channels, padding=0)
        
    def forward(self,x):
        out = self.conv_Gx(self.padx(x))
        out = self.conv_Gy(self.pady(out))
        out = self.conv_Gz(self.padz(out))
        return out
    
class ReplicationPad(nn.Module):
    
    def __init__(self, pad):
        super(ReplicationPad, self).__init__()
        if len(pad) == 6:
            self.pad = nn.ReplicationPad3d(list(reversed(pad)))
        if len(pad) == 4:
            self.pad = nn.ReplicationPad2d(list(reversed(pad)))
        if len(pad) == 2:
            self.pad = nn.ReplicationPad1d(list(reversed(pad)))
    
    def forward(self,x):
        return self.pad(x)
    
class Gradient3D(nn.Module):
    
    def __init__(self, input_channels):
        super(Gradient3D, self).__init__()
        dx = kernels3D['dx']()
        dy = kernels3D['dy']()
        dz = kernels3D['dz']()
        self.padx = nn.ReplicationPad3d(list(reversed([1,1,0,0,0,0])))
        self.pady = nn.ReplicationPad3d(list(reversed([0,0,1,1,0,0])))
        self.padz = nn.ReplicationPad3d(list(reversed([0,0,0,0,1,1])))
        self.conv_dx = Convolve(dx, input_channels, padding=0)
        self.conv_dy = Convolve(dy, input_channels, padding=0)
        self.conv_dz = Convolve(dz, input_channels, padding=0)
        
    def forward(self,x):
        dx = self.conv_dx(self.padx(x))
        dy = self.conv_dy(self.pady(x))
        dz = self.conv_dz(self.padz(x))
        return dx, dy, dz
    
class DFDT(th.nn.Module):
    def __init__(self, input_channels, t=5):
        super(DFDT, self).__init__()
        self.conv_g = GaussianBlur3D(t, input_channels)
        self.conv_n = Gradient3D(input_channels)
        self.conv_nt = Gradient3D(1)
        
    def forward(self, inputs):
        x, phi_t0 = inputs
        phi_t = self.conv_g(1-phi_t0)
        im_dx_t, im_dy_t, im_dz_t = self.conv_n(x)
        phi_t_dx, phi_t_dy, phi_t_dz = self.conv_nt(phi_t)
        eps = 1e-10
        grad_phi_t = th.sqrt(phi_t_dx**2 + phi_t_dy**2 + phi_t_dz**2 + eps) 
        dI_dt = ( im_dx_t*phi_t_dx + im_dy_t*phi_t_dy + im_dz_t*phi_t_dz )/grad_phi_t
        return dI_dt

##################################################### TRAINABLE ##################################################### TRAINABLE 
    
class MeanGPool(nn.Module):
    
    def __init__(self):
        super(MeanGPool, self).__init__()
        
    def forward(self,x):
        x = x.view(*x.shape[0:2],-1).mean(-1)
        return x
    
class StdGPool(nn.Module):
    
    def __init__(self):
        super(StdGPool, self).__init__()
        
    def forward(self,x):
        x = x.view(*x.shape[0:2],-1).std(-1)
        return x
    
class ExpandGFunction(nn.Module):    
    
    def __init__(self, merge=True):
        super(ExpandGFunction, self).__init__()
        self.pre_forward = self.expand_forward
        self.pre_forward = self.merge_forward if merge else self.pre_forward
        
    def expand_forward(self,x_l,x_g):
        spatial_dims = x_l.shape[2:]
        bs,ch_g = x_g.shape
        x_g = x_g.unsqueeze(-1).unsqueeze(-1)
        x_g = x_g.expand(-1,-1,*spatial_dims)
        return x_g
        
    def merge_forward(self,x_l,x_g):
        x_g = self.expand_forward(x_l,x_g)
        x = torch.cat((x_l,x_g),1)
        return x
        
    def forward(self,x_l,x_g):
        return pre_forward(x_l,x_g)

    
class GFunction(nn.Module):
    """
    Implement spatial invaiant feature.
    This Gfunction are invariant to permutation of pixels therefore
    they are invariant to spatial transformation/deformation of the input.
    
    Mixing these features with transform variant features makes them variant aswell
    however it can be shown that they perform similar to histog
    
    Arguments:
    in_channels: int
        input_channels.
    global_channels: int
        output channels.
    stride: int
        stride of convolution.
    groups: int
        depthwise convolution factor.
    kernel_size: int
        size of convolution kernel.
    bias: bool
        if bias is added to convolution output.
    activatiton: nn.Module
        activation function.
    mode: str
        {avg, std, avg_std} define the type of global pooling function.
        in case avg given an input of bs,in_channels,h,w,d the output is
        bs,in_channels+global_channels,h,w,d in case of avg_std 
        bs,in_channels+global_channels*2,h,w,d.
    """
    def __init__(self, in_channels, global_channels, stride=1, groups=1, kernel_size=1, 
                 bias=True, activation=nn.LeakyReLU(inplace=True),
                 mode='avg', padding='same', merge=True, conv_mode='3d'):
        
        self.conv = Conv(in_channels, global_channels, 1, stride=stride,
                           padding=padding, groups=groups_in, bias=bias, mode=conv_mode)
        self.act = activation
        self.mean_pool = MeanGPool()
        self.std_pool = StdGPool()
        self.expander = ExpandGFunction(merge=merge)

        # instruction prefetch
        self.pre_forward = self.forward_avg_mode
        self.pre_forward = self.forward_avg_mode if mode == 'std' else self.pre_forward
        self.pre_forward = self.forward_avg_std_mode if mode == 'avg_std' else self.pre_forward
        
    def forward_avg_mode(self, x):
        return self.mean_pool(x)
    
    def forward_std_mode(self, x):
        return self.std_pool(x)
    
    def forward_avg_std_mode(self, x):
        return torch.cat((self.forward_std_mode(x),
                          self.forward_avg_mode(x)),1)
        
    def forward(self, x):
        return self.expander(self.pre_forward(x),x)
    
    
class UpSample(nn.Module):
    """
    Implementing generic upsampling strategy
    Parameters:
    up_scale: int
        as a result spatial dimensions of the output will be scaled by up_scale.
    """
    def __init__(self, up_scale=2, mode='3d'):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=up_scale, mode='trilinear' if mode =='3d' else 'bilinear')
        
    def forward(self, x):
        return self.up(x)
    
class MaxPooling(nn.Module):
    
    def __init__(self, down_scale=2, kernel_size=2, mode='3d'):
        super(MaxPooling, self).__init__()
        if mode=='2d':
            self.down = nn.MaxPool2d(kernel_size=kernel_size, stride=down_scale)
        if mode=='3d':
            self.down = nn.MaxPool3d(kernel_size=kernel_size, stride=down_scale)
        
    def forward(self, x):
        return self.down(x)
    
class UpConv(nn.Module):
    """
    Implementing generic upsampling strategy
    Parameters:
    up_scale: int
        as a result spatial dimensions of the output will be scaled by up_scale.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1, groups=1, up_scale=2, bias=True, mode='3d'):
        super(UpConv, self).__init__()
        padding_in = compute_upconv_input_padding(2,4,up_scale,1,2*up_scale,output_padding=0)
        conv_type = nn.ConvTranspose3d if mode == '3d' else nn.ConvTranspose2d
        self.up = conv_type(in_channels, out_channels, kernel_size=2*up_scale, dilation=dilation,
                            stride=up_scale, padding=padding_in)
            
    def forward(self, x):
        x = self.up(x)
        return x
    
class Conv(nn.Module):
    """
    Wrapping Conv with advanced padding options.
    args:
        in_channels, out_channels, kernel_size, dilation, 
        groups, stride, bias: look at torch.nn.Conv3d
        padding: string {same, valid} same: input and output have the same shape, 
                                      valid: no zero padding.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=1, bias=True,
                 stride=1, padding='same', mode='3d'):
        super(Conv, self).__init__()
        conv_type = nn.Conv3d if mode =='3d' else nn.Conv2d
        self.layer = conv_type(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                               dilation=dilation, groups=groups, bias=bias)
        self.stride, self.dilation, self.padding, self.kernel_size = stride, dilation, padding, kernel_size

    def forward(self, x):
        input_shape = list(x.shape)[2:]
        pad_func, output_padding = conv_padding_func(self.padding, input_shape, 
                                                     self.kernel_size, self.dilation, self.stride)
        return self.layer(pad_func(x))

class Conv3D(Conv):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, 
                 groups=1, bias=True, stride=1, padding='same'):
        super(Conv3D, self).__init__( in_channels, out_channels, kernel_size=kernel_size, 
                                      dilation=dilation, groups=groups, bias=bias,
                                      stride=stride, mode='3d' )
    
class Conv2D(Conv):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, 
                 groups=1, bias=True, stride=1, padding='same'):
        super(Conv2D, self).__init__( in_channels, out_channels, kernel_size=kernel_size, 
                                      dilation=dilation, groups=groups, bias=bias,
                                      stride=stride, mode='2d' )
           
class ConvBlock(nn.Module):
    """
    ConvBlock3D is the main building block of a convolutional neural network, it is parametrized to cover
    most use cases for conv nets like resblocks and batch norm.
    It mainly consists of a series of convolutions each followed by an activation function.
        
    the first convolution transforms:
        R^in_channels->R^out_channels
    the num_blocks - 1 following convolutions transforms: 
        R^out_channels->R^out_channels.
    
    Parametrizations:
    - residual and batch_norm are unset:
    >---|conv_0|act|---...---|conv_i|act|---...---|conv_{num_blocks-1}|act|--->
                                
    - residual is unset and batch_norm is set:
    >---|conv_0|act|---...---|conv_{num_blocks-1}|act|---|batch_norm|--->
    
    - residual is set and batch_norm is unset:
    >---|conv_0|act|---...---|conv_i|act|---...---|conv_{num_blocks-1}|act|-(+)->
                    |__...__________________...______________________________| 
           
    - residual and batch_norm are set:
    >---|conv_0|act|---...---|conv_i|act|---...---|conv_{num_blocks-1}|act|-(+)-|batch_norm|--->
                    |__...__________________...______________________________| 
    Attributes:
    act: nn.Module 
        activation funtion.
    initial_conv: nn.Conv3D
        corresponds to conv_0.
    convs: nn.ModuleList
        list of convolutions layers.
    bns: nn.ModuleList
        list of batch norm layers.
        
    Parameters:
    num_convs: int
        number of convolution in this block
    in_channels: int
        number of input channels in this block
    out_channels: int
        number of output channels in this block
    kernel_size: {int, tuple of int}
        size of convolution kernel.
    stride:  {int, tuple of int}
        sampling step of convolution function, each convolution is computed each stride pixels.
    dilatation:  {int, tuple of int}
        implement atrous convolution, a trick to have bigger support and sparse kernels.
    groups_in: int
        depth wise convolution of conv 0
    groups_out: int
        depth wise convolution of conv 1,...,num_convs-1
    depth_initialization: int
        use fixup initialization starting at depth depth_initialization.
    bias: bool
        use bias in conv.
    residual: bool
        use residual scheme.
    batch_norm: bool
        use batch norm residual scheme.
    padding: string 
        {'same','valid'} padding schemes.
    activation: nn.Module
        activation to be used.
    """
    def __init__(self, num_convs, in_channels, out_channels, 
                 kernel_size=3, stride=1, dilation=1, groups_in=1, groups_out=1, init_depth=0, 
                 bias=True, residual=True, batch_norm=True, conv_mode='3d',
                 padding='same', activation=nn.LeakyReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        
        # activation   
        self.act = activation
        
        # conv input_channels -> output_channels
        self.initial_conv = Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                                 dilation=dilation, groups=groups_in, bias=bias, mode=conv_mode)
        # convs output_channels -> output_channels 
        self.convs = nn.ModuleList([Conv(out_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                                         dilation=dilation, groups=groups_out, bias=bias, mode=conv_mode)] * (num_convs - 1))
        
        # depth fixup initialization arXiv:1901.09321v2
        if init_depth>0:
            self.initial_conv.apply(depth_scale_weights(init_depth))
            for i,conv in enumerate(self.convs): conv.apply(zero_weights())
        
        # batch normalization
        # bn output_channels * (num_blocks-1)
        if batch_norm: 
            self.bn  = nn.BatchNorm3d(out_channels) if conv_mode=='3d' else nn.BatchNorm2d(out_channels)
        
        # instruction prefetch
        self.pre_forward = self.vanilla_forward
        self.pre_forward = self.residual_forward if residual else self.pre_forward
        self.pre_forward = self.batch_norm_forward if batch_norm else self.pre_forward
        self.pre_forward = self.residual_batch_norm_forward if residual and batch_norm else self.pre_forward
        
    def batch_norm_forward(self,x):
        x = self.vanilla_forward(x)
        return self.bn(x)
        
    def residual_batch_norm_forward(self,x):
        x = self.residual_forward(x)
        return self.bn(x)

    def residual_forward(self, x):
        identity_x = x
        for conv in self.convs: x = self.act(conv(x))
        return x + identity_x

    def vanilla_forward(self, x):
        for conv in self.convs: x = self.act(conv(x))
        return x

    def forward(self, x):
        x = self.act(self.initial_conv(x))
        return self.pre_forward(x)
    
class ConvBlock3D(ConvBlock):
    def __init__(self, num_convs, in_channels, out_channels, 
                 kernel_size=3, stride=1, dilation=1, groups_in=1, groups_out=1, init_depth=0, 
                 bias=True, residual=True, batch_norm=True,
                 padding='same', activation=nn.LeakyReLU(inplace=True)):
        
        super(ConvBlock3D,self).__init__(num_convs, in_channels, out_channels, kernel_size=kernel_size,
                                         stride=stride, dilation=dilation, groups_in=groups_in, groups_out=groups_out,
                                         init_depth=init_depth, bias=bias, residual=residual, batch_norm=batch_norm,
                                         conv_mode='3d', padding=padding, activation=activation)
        
class ConvBlock2D(ConvBlock):
    def __init__(self, num_convs, in_channels, out_channels, 
                 kernel_size=3, stride=1, dilation=1, groups_in=1, groups_out=1, init_depth=0, 
                 bias=True, residual=True, batch_norm=True,
                 padding='same', activation=nn.LeakyReLU(inplace=True)):
        
        super(ConvBlock2D,self).__init__(num_convs, in_channels, out_channels, kernel_size=kernel_size,
                                         stride=stride, dilation=dilation, groups_in=groups_in, groups_out=groups_out,
                                         init_depth=init_depth, bias=bias, residual=residual, batch_norm=batch_norm,
                                         conv_mode='2d', padding=padding, activation=activation)