import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self,features:int, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def froward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        std = x.std(dim =-1,keepdim = True )
        return self.alpha*(x-mean)/(std+self.eps)+self.bias 

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float) ->None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size: int) ->None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    