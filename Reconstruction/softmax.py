import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from statistics import mean
import numpy as np
import wandb
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import init_weights_vit_timm, _load_weights, init_weights_vit_jax
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from utils import named_apply
import copy
 
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., layerth=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.layerth = layerth
        head_dim = dim // num_heads
        # sqrt (D)
        self.scale = head_dim ** -0.5
        self.reconstruct = False
        self.test = False
        self.train_error = 0
        self.test_error = 0

    
        self.train_rel_error_wrt_h = 0
        self.train_rel_error_wrt_phiq = 0
        self.test_rel_error_wrt_h = 0
        self.test_rel_error_wrt_phiq = 0
        
        self.train_h_norms_ave = 0
        self.train_phiq_norms_ave = 0
        self.test_h_norms_ave = 0
        self.test_phiq_norms_ave = 0
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B, heads, N, features
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # normalise q,k
        q = (q - q.mean(dim=-1, keepdim=True)) / q.std(dim=-1, keepdim=True) # they were closed until the last run
        k = (k - k.mean(dim=-1, keepdim=True)) / k.std(dim=-1, keepdim=True)  # they were closed until the last run
        qk = (q @ k.transpose(-2, -1)) * self.scale # [B, heads, N, N]
        attn = qk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        # @ is a matrix multiplication
        x = (attn @ v)
        
        # don't do it in each forward pass
        # self._compare_v_vhat(q, k, v) # it updates the lists in the class
        if hasattr(self, 'calculate_kpca') and self.calculate_kpca:
            self.last_q = q
            self.last_k = k
            self.last_v = v
        
                    
        if self.reconstruct:   
            qk = q @ k.transpose(-2, -1) * self.scale # [B, heads, N, N]
            qq = q @ q.transpose(-2, -1) * self.scale # [B, heads, N, N]
            
            # e^(q_i^T q_i / √D)
            num = torch.exp(torch.diagonal(qq, dim1=-2, dim2=-1)) # B,H,N
            # sum( e^(q_i^T k_i / √D))^2
            den = torch.exp(qk).sum(dim=-1).pow(2)  # B,H,N
            
            phiq_norm = num/den
            # B, heads, N, [C/heads]
            h_norm = x.pow(2).sum(dim=-1)  # B,H,N           
            error = phiq_norm - h_norm
            abs_error = error.abs()
            rel_error_wrt_h = (abs_error / (h_norm + 1e-8)).mean()
            rel_error_wrt_phiq = (abs_error / (phiq_norm + 1e-8)).mean() 
                        
            # take absolute value and then mean of errors
            error_ave = abs_error.mean()
            
          
            if self.test:
                self.test_error=error_ave
                self.test_rel_error_wrt_h = rel_error_wrt_h
                self.test_rel_error_wrt_phiq = rel_error_wrt_phiq
                self.test_h_norms_ave = h_norm.mean()
                self.test_phiq_norms_ave = phiq_norm.mean()
            else:
                self.train_error=error_ave
                self.train_rel_error_wrt_h = rel_error_wrt_h
                self.train_rel_error_wrt_phiq = rel_error_wrt_phiq
                self.train_h_norms_ave = h_norm.mean()
                self.train_phiq_norms_ave = phiq_norm.mean()

            self.test = False
            self.reconstruct = False

        x = x.transpose(1, 2).reshape(B,N,C)
       
        x = self.proj(x)
        x = self.proj_drop(x)
        ################ COSINE SIMILARITY MEASURE
        # n = x.shape[1] #x is in shape of (batchsize, length, dim)
        # sqaure norm across features
        # x_norm = torch.norm(x, 2, dim = -1, keepdim= True)
        # x_ = x/x_norm
        # x_cossim = torch.tril((x_ @ x_.transpose(-2, -1)), diagonal= -1).sum(dim = (-1, -2))/(n*(n - 1)/2)
        # x_cossim = x_cossim.mean()
        # python debugger breakpoint
#         import pdb;pdb.set_trace()
        ################
       
        return x


class Block(nn.Module):
 
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerth = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop,layerth=layerth)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layerth = layerth
 
    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
 
 
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
 
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',pretrained_cfg=None,pretrained_cfg_overlay=None,wandb=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.wandb = wandb
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.reconstruct = False
        self.test_check = False
        self.depth = depth
       
        # how does embedding conv2d update its weights? 
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # print(img_size,patch_size,in_chans,num_patches)
 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                layerth = i)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
 
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([f
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
 
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
        self.init_weights(weight_init)
 
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            partial(init_weights_vit_jax(mode, head_bias), head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            init_weights_vit_timm
 
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights(m)
 
    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
 
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
 
    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist
 
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # add the same pos_emb token to each sample? broadcasting...
        x = self.pos_drop(x + self.pos_embed)
        if self.reconstruct:
            for i in range(0,self.depth):
                self.blocks[i].attn.reconstruct=True
        x = self.blocks(x)
        x = self.norm(x)
                
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
 
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        train_error = []
        test_error = []
        
        train_rel_error_wrt_h = []
        train_rel_error_wrt_phiq = []
        test_rel_error_wrt_h = []
        test_rel_error_wrt_phiq = []
        
        if self.reconstruct and self.test_check:
            for i in range(0,self.depth):
                test_error.append(self.blocks[i].attn.test_error.data.cpu())
                test_rel_error_wrt_h.append(self.blocks[i].attn.test_rel_error_wrt_h.data.cpu())
                test_rel_error_wrt_phiq.append(self.blocks[i].attn.test_rel_error_wrt_phiq.data.cpu())
                test_h_norms_ave.append(self.blocks[i].attn.test_h_norms_ave.data.cpu())
                test_phiq_norms_ave.append(self.blocks[i].attn.test_phiq_norms_ave.data.cpu())
                
            if self.wandb:
                wandb.log({"test_error":np.average(test_error)})
                wandb.log({"test_rel_error_wrt_h":np.average(test_rel_error_wrt_h)})
                wandb.log({"test_rel_error_wrt_phiq":np.average(test_rel_error_wrt_phiq)})
                wandb.log({"test_h_norms_ave":np.average(test_h_norms_ave)})
                wandb.log({"test_phiq_norms_ave":np.average(test_phiq_norms_ave)})
        elif self.reconstruct:
            for i in range(0,self.depth):
                train_error.append(self.blocks[i].attn.train_error.data.cpu())
                train_rel_error_wrt_h.append(self.blocks[i].attn.train_rel_error_wrt_h.data.cpu())
                train_rel_error_wrt_phiq.append(self.blocks[i].attn.train_rel_error_wrt_phiq.data.cpu())
                train_h_norms_ave.append(self.blocks[i].attn.train_h_norms_ave.data.cpu())
                train_phiq_norms_ave.append(self.blocks[i].attn.train_phiq_norms_ave.data.cpu())
                
            if self.wandb:
                wandb.log({"train_error":np.average(train_error)})
                wandb.log({"train_rel_error_wrt_h":np.average(train_rel_error_wrt_h)})
                wandb.log({"train_rel_error_wrt_phiq":np.average(train_rel_error_wrt_phiq)})
                wandb.log({"train_h_norms_ave":np.average(train_h_norms_ave)})
                wandb.log({"train_phiq_norms_ave":np.average(train_phiq_norms_ave)})
        return x
 
    def test(self):
        for i in range(0,self.depth):
            self.blocks[i].attn.test=True



