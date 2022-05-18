# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict

import copy
from typing import Optional, List

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


#-----------------------------------------------------------------------------
# KAUSHIK ADDED THE FOLLOWING

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



#---------------------------------------

# BOTNET

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)
        
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




#-----------------------------------------------------------------------------

class TransPoseR_alpha(nn.Module):

    def __init__(self, block, layers, w, h, pos_embedding_type, n_head, encoder_layers_num, dim_feedforward, d_model):
        self.inplanes = 64
        super(TransPoseR_alpha, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        #self.layer2 = self._make_layer(block, 96, layers[1], stride=2, resolution=[64,48])
        #self.layer3 = self._make_layer(block, 128, layers[2], stride=1, resolution=[32,24])
        #self.layer4 = self._make_layer(block, 256, layers[3], stride=1, heads=4, mhsa=True, resolution=[32,24])
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, resolution=[64,48])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, resolution=[32,24])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, heads=4, mhsa=True, resolution=[32,24])
        
        #self.layer4 = self._make_layer(block, 256, layers[3], stride=1, heads=8, mhsa=True, resolution=[32,24])
        # NOTE: RESNET USES 64, 128, 256, 512


        #d_model: 256 
        #dim_feedforward: 1024 
        #encoder_layers_num: 4 
        #n_head: 8 
        #pos_embedding_type: learnable or sine 
        #w: 192 
        #h: 256
        
        self.reduce = nn.Conv2d(self.inplanes, d_model, 1, bias=False)
        self._make_position_embedding(w, h, d_model, pos_embedding_type)
        
        #------------------------------------ 
        
        '''
        self.to_patch_embedding1 = self.get_patch_embedding(patch_size=16, in_channels=3, dim=d_model)
        
        num_patches = 192 #768 
        self.pos_embedding_kb1 = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.dropout_kb1 = nn.Dropout(0.0)
        dim_head = 64
        self.vit_kb1 = Transformer(dim=d_model, depth=4, heads=8, mlp_dim=1024, dim_head=dim_head)
        self.mlp_kb1 = nn.Linear(num_patches, 768)

        self.conv1x1 = nn.Conv2d(
            in_channels=512, 
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0
        )
        '''
        
        #------------------------------------

        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
            return_atten_map=False,
            pe_only_at_begin=True # KAUSHIK ADDED THIS LINE
        )
         

        
    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
        else:
            assert pe_type == 'learnable' # KAUSHIK ADDED THIS
            
            with torch.no_grad():
                self.pe_h = h // 8
                self.pe_w = w // 8
                length = self.pe_h * self.pe_w
                
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
   

    '''
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    '''

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False, resolution=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.inplanes, planes, stride, heads, mhsa, resolution))
            if stride == 2:
                resolution[0] /= 2
                resolution[1] /= 2
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers) 

    def get_patch_embedding(self, patch_size, in_channels, dim):
        patch_dim = in_channels * patch_size * patch_size
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )    

   
    def forward(self, x):
        # x: torch.Size([20, 3, 256, 192])

        #-------------
        '''
        x1 = x
        x2a = self.to_patch_embedding1(x1)
        
        x2 = x2a #torch.cat([x2a, x2b], dim=1)
        # x2: torch.Size([20, 192, 256])
        
        bs, n, _ = x2.shape
        x2 += self.pos_embedding_kb1[:, :n]
        
        x2 = self.dropout_kb1(x2)
        
        x2 = self.vit_kb1(x2)
        # x2: torch.Size([20, 192, 256])
        
        x2 = x2.permute(0, 2, 1)
        
        x2 = F.relu(self.mlp_kb1(x2))
        # x2: torch.Size([20, 256, 768])
        
        x2 = x2.permute(2, 0, 1)
        # x2: torch.Size([768, 20, 256])
        
        x2 = x2.permute(1, 2, 0).contiguous().view(bs, 256, 32, 24)
        # x2: torch.Size([20, 256, 32, 24])
        '''
        #-------------
        
        x = self.conv1(x) # 7x7 conv with 64 feature maps
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x: torch.Size([20, 64, 64, 48])

        x = self.layer1(x)
        # x: torch.Size([20, 256, 64, 48])
        
        x = self.layer2(x)
        # x: torch.Size([20, 384, 32, 24])
        
        x = self.layer3(x)
        # x: torch.Size([16, 512, 32, 24])
        
        x = self.layer4(x)
        # x: torch.Size([16, 1024, 32, 24])

        x = self.reduce(x) # 1x1 conv
        # x: torch.Size([20, 256, 32, 24])
        
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        # x: torch.Size([768, 20, 256])
       
        x = self.global_encoder(x, pos=self.pos_embedding)
        # x: torch.Size([768, 20, 256])
        
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # x: torch.Size([20, 256, 32, 24])
        
        
        #----------------------------------------------------------
        
        '''
        # KAUSHIK ADDED THE FOLLOWING LINE
        x = torch.cat([x, x2], dim=1)
        # x: torch.Size([20, 512, 32, 24])
        
        x = self.conv1x1(x)
        # torch.Size([20, 256, 32, 24])
        '''
                                      
        return x

    def init_weights(self, pretrained=''):
        # init with normal dist. weights; then load
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
    
    
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            existing_state_dict = {}
            for name, m in pretrained_state_dict['teacher'].items():
                if "module.backbone" in name:
                    name_ = name.replace('module.backbone.', '')
                    if name_ in self.state_dict():
                        existing_state_dict[name_] = m
                        print(f"{name_} is loaded from {pretrained}")
            self.load_state_dict(existing_state_dict, strict=False)
        else:
            print('KAUSHIK: unable to load pretrained weights! ')
            #assert 1 == 2
            
            


# KAUSHIK ADDED THE ABOVE

#-----------------------------------------------------------------------------

class TransPoseR(nn.Module):

    def __init__(self, model_alpha, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(TransPoseR, self).__init__()
        self.model_alpha = model_alpha
        
        d_model = cfg.MODEL.DIM_MODEL
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        encoder_layers_num = cfg.MODEL.ENCODER_LAYERS
        n_head = cfg.MODEL.N_HEAD
        pos_embedding_type = cfg.MODEL.POS_EMBEDDING
        w, h = cfg.MODEL.IMAGE_SIZE

        #d_model: 256 
        #dim_feedforward: 1024 
        #encoder_layers_num: 4 
        #n_head: 8 
        #pos_embedding_type: learnable or sine 
        #w: 192 
        #h: 256
        
        
        # used for deconv layers
        self.inplanes = d_model
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,   # 1
            extra.NUM_DECONV_FILTERS,  # [d_model]
            extra.NUM_DECONV_KERNELS,  # [4]
        )

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'


        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
     
        x = self.model_alpha(x)
        #x = F.relu(x)
        
        x = self.deconv_layers(x)
        # x: torch.Size([20, 256, 64, 48])
        
        x = self.final_layer(x)
        # x: torch.Size([20, 17, 64, 48])

        return x

    def init_weights(self, pretrained=''):
        # init with normal dist. weights; then load
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
    
    
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            existing_state_dict = {}
            for name, m in pretrained_state_dict['teacher'].items():
                if "module.backbone" in name:
                    name_ = name.replace('module.backbone.', '')
                    name_ = "model_alpha." + name_
                    if name_ in self.state_dict():
                        existing_state_dict[name_] = m
                        print(f"{name_} is loaded from {pretrained}")
            self.load_state_dict(existing_state_dict, strict=False)
        else:
            print('KAUSHIK: unable to load pretrained weights! ')
            #assert 1 == 2


#------------------------

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
  
#------------------------    


'''
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}
'''


def get_pose_net(cfg, is_train, **kwargs):
    print('pretrained: ', cfg.MODEL.PRETRAINED)
    
    layers = [3, 4, 6, 3]
    
    #model_alpha = TransPoseR_alpha(block_class, layers, cfg, **kwargs)
    model_alpha = TransPoseR_alpha(Bottleneck, layers, w=192, h=256, pos_embedding_type='learnable', n_head=8, encoder_layers_num=4, dim_feedforward=1024, d_model=256)
    #model_alpha = TransPoseR_alpha(Bottleneck, layers, w=192, h=256, pos_embedding_type='learnable', n_head=8, encoder_layers_num=6, dim_feedforward=1024, d_model=256)
    
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model_alpha.init_weights(cfg.MODEL.PRETRAINED)
    print('1 ====')
    
    model = TransPoseR(model_alpha, Bottleneck, layers, cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    print('2 ====')

    print('num of params: ', get_n_params(model)/(1e6), 'M')  

    return model
    
    
