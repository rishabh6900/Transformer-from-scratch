import troch 
import troch.nn as nn 
import math 

class LayerNormalization(nn.Module):
    def __init__(self,feature:int, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(feature))