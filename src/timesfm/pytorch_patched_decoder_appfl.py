import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pytorch_patched_decoder as ppd
config = ppd.TimesFMConfig()

class TimesFmAppfl(nn.Module):

    def __init__(self, lookback: int = 512, lookahead: int = 96, context_len: int = 512):

        super(TimesFmAppfl, self).__init__()
        
        self.timesfm = ppd.PatchedTimeSeriesDecoder(config)
        self.lookback, self.lookahead = lookback, lookahead
        self.context_len = context_len

    def load_state_dict(self, state_dict):

        return self.timesfm.load_state_dict(state_dict)

    def state_dict(self):

        return self.timesfm.state_dict()
    
    def pad_tensor(self, x):

        B, L = x.shape
        device = x.device
        dtype = x.dtype
        
        if L < self.context_len:
            padded_input = torch.zeros((B, self.context_len), device=device, dtype=dtype)
            padded_input[:, -L:] = x
            padding = torch.ones((B, self.context_len), device=device, dtype=dtype)
            padding[:, -L:] = 0
        else:
            padded_input = x[:, -self.context_len:]
            padding = torch.zeros((B, self.context_len), device=device, dtype=dtype)
        
        freq = torch.zeros((B, 1), device=device, dtype=torch.long)
        
        return padded_input, torch.cat((padding,torch.zeros((B,self.lookahead),device=device,dtype=dtype)),dim=-1), freq
    
    def forward(self, x):

        padded_inp, padding, freq = self.pad_tensor(x)
        return self.timesfm.decode(padded_inp,padding,freq,self.lookahead)[0] # ignoring quantiles