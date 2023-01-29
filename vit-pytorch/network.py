'''
Vision Transformer (ViT) in PyTorch
Created by: Mitterrand Ekole
Based on the paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
Date: 1/29/2023
'''
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

#fxn to split image into patches in pytorch

def split_image_into_patches(image, patch_size):  
    image=Image.open(image)
    transform=transforms.Compose([transforms.ToTensor()])
    img_tensor=transform(image)
    kernel_size, stride = patch_size, patch_size
    patches = img_tensor.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)

    #print(patches.shape)
    return patches

''' Multihead attention layer for ViT'''

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim=embed_dim
        self.n_heads=n_heads
        self.head_dim=embed_dim//n_heads

        self.q_mapper=nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for __ in range(n_heads)])
        self.k_mapper=nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for __ in range(n_heads)])
        self.v_mapper=nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for __ in range(n_heads)])
        self.softmax=nn.Softmax(dim=-1)


    def forward(self, sequences):
        #seq shape: (batch_size, seq_len, embed_dim) 
        #Go into (batch_size, seq_len, n_heads, head_dim)
        #return to (batch_size, seq_len, embed_dim) via concatenation

        result=[]
        for sequence in sequences:
            seq_result=[]
            for head in range(self.n_heads):
                q_mapping=self.q_mapper[head]
                k_mapping=self.k_mapper[head]
                v_mapping=self.v_mapper[head]

                seq=sequence[:,head*self.head_dim:(head+1)*self.head_dim]
                q,k,v=q_mapping(seq),k_mapping(seq),v_mapping(seq)

                attention=self.softmax(torch.matmul(q,k.transpose(1,2)))
                seq_result.append(torch.matmul(attention,v))
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsuqeeze(x,0) for x in result],dim=0)

'''ViT Encoder layer'''

class EncoderLayer(nn.Module):
    def __init__(self,embed_dim, n_heads, mlp_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention=MultiHeadAttention(embed_dim, n_heads) #multihead attention
        self.norm1=nn.LayerNorm(embed_dim)  #normalization layer
        self.mlp=nn.Sequential( #MLP
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2=nn.LayerNorm(embed_dim)

    def forward(self, x):
        out=x+self.attention(self.norm1(x))
        out=out+self.mlp(self.norm2(out))
        return out

'''ViT Model'''

def get_positional_embeddings(image_size, patch_size, embed_dim):
    #image_size=c*h*w, patch_size=number of patches
    patch=(image_size[1]//patch_size, image_size[2]//patch_size)
    position_embeddings=torch.zeros(patch[0]*patch[1]+1, embed_dim)
    position=torch.arange(0, patch[0]*patch[1]+1).unsqueeze(1)
    div_term=torch.exp(torch.arange(0, embed_dim, 2)*(-np.log(10000.0)/embed_dim))
    position_embeddings[:, 0::2]=torch.sin(position*div_term)
    position_embeddings[:, 1::2]=torch.cos(position*div_term)
    return position_embeddings

class ViT(nn.Module):
    def __init__(self, image_size,patch_size,n_blocks,embed_dim=8,n_heads=2, out_d=10):
        super(ViT, self).__init__()

        # image_size=c*h*w, patch_size=number of patches
        self.patch_size=patch_size
        self.image_size=image_size
        self.n_blocks=n_blocks
        self.embed_dim=embed_dim
        self.n_heads=n_heads

        #checking for input and patch size

        assert image_size[1] % patch_size == 0, "Input size must  not be divisible by number of patches"
        assert image_size[2] % patch_size == 0, "Input size must not be divisible by number of patches"

        self.patch=(image_size[1]//patch_size, image_size[2]//patch_size)


        #Linear mapper

        self.input_d=int(image_size[0]*self.patch[0]*self.patch[1])
        self.linear_mapper=nn.Linear(self.input_d, embed_dim)

        #Learnable token -- to be modified later
        self.class_token=nn.Parameter(torch.randn(1,1,embed_dim))
        self.register_buffer("position_embeddings", get_positional_embeddings(image_size, patch_size, embed_dim))
        self.blocks=nn.ModuleList([EncoderLayer(embed_dim, n_heads, embed_dim*4, 0.1) for __ in range(n_blocks)])
        self.mlp=nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.Softmax(dim=-1),
        )

    def forward(self, images):
        patches=split_image_into_patches(images, self.patch_size).to(self.position_embeddings.device)
        #run linear layer tokenization and positional embeddings

        tokens=self.linear_mapper(patches)
        tokens=torch.cat([self.class_token.expand(tokens.shape[0],-1,-1), tokens], dim=1)

        #add positional embeddings
        out=tokens+self.position_embeddings.repeat(tokens.shape[0],1,1)

        #tranformer blocks
        for block in self.blocks:
            out=block(out)

        #return classification token only--to be modified later for SLAM

        out=out[:,0]
        return self.mlp(out)




            