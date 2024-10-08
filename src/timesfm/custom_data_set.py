import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Dict

class NRELComstock(Dataset):
    
    def __init__(
        self,
        data_array: np.ndarray,
        num_bldg: int = 12,
        lookback: int = 12,
        lookahead:int = 4,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        context_len: int = 512,
        output_unpadded_inputs: bool = False
    ):
        
        if data_array.shape[0] < num_bldg:
            raise ValueError('More buildings than present in file!')
        else:
            self.data = data_array[:num_bldg,:,:]
        self.num_clients = num_bldg
        
        # lookback and lookahead
        self.lookback, self.lookahead, self.context_len = lookback, lookahead, context_len

        # ensure presence of single feature
        self.data = self.data[:,:,0] # ensure that only the first dimension is chosen 
        
        # calculate statistics
        stacked = self.data.reshape(-1)
        if (mean is None) or (std is None): # statistics are not provided. Generate it from data
            self.mean = stacked.mean()
            self.std = stacked.std()
        else: # statistics are provided. Use the provided statistics
            self.mean = mean
            self.std = std
            
        self.ndata = (self.data-self.mean)/self.std # normalized data
        
        # disambiguating between clients
        len_per_client = self.data.shape[1] - lookback - lookahead + 1
        self.total_len = len_per_client * self.num_clients
        
        # save whether to normalize, and the data type
        self.output_unpadded_inputs = output_unpadded_inputs
        self.normalize = normalize
        self.dtype = dtype
        
        # if normalization is disabled, return to default statistics
        if not self.normalize:
            self.mean = np.zeros_like(self.mean)
            self.std = np.ones_like(self.std)

        # generate the padding indicators
        self._generate_padding()
        self._generate_freq_indicator()
        
    def _client_and_idx(self, idx):
        
        part_size = self.total_len // self.num_clients
        part_index = idx // part_size
        relative_position = idx % part_size
        
        return relative_position, part_index
    
    def __len__(self):
        
        return self.total_len

    def _generate_padding(self):

        self.padding = np.zeros((self.context_len+self.lookahead,))
        padded_len = max(0,self.context_len-self.lookback)
        self.padding[:padded_len] = 1

    def _generate_freq_indicator(self):

        self.freq = np.zeros((1,))
    
    def __getitem__(self, idx):
        
        tidx, cidx = self._client_and_idx(idx)
        
        x = np.zeros((self.context_len,))
        if self.normalize:
            x[-self.lookback:] = self.ndata[cidx,tidx:tidx+self.lookback][-self.context_len:]
            y = self.ndata[cidx,tidx+self.lookback:tidx+self.lookback+self.lookahead][-self.context_len:]
            if self.output_unpadded_inputs:
                x_unpadded = self.ndata[cidx,tidx:tidx+self.lookback]
        else:
            x[-self.lookback:] = self.ndata[cidx,tidx:tidx+self.lookback][-self.context_len:]
            y = self.data[cidx,tidx+self.lookback:tidx+self.lookback+self.lookahead]
            if self.output_unpadded_inputs:
                x_unpadded = self.ndata[cidx,tidx:tidx+self.lookback]
            
        # convert to pytorch format
        x,y,padding,freq = torch.tensor(x,dtype=self.dtype), torch.tensor(y,dtype=self.dtype), torch.tensor(self.padding,dtype=self.dtype), torch.LongTensor(self.freq)
        if self.output_unpadded_inputs:
                x_unpadded = torch.tensor(x_unpadded, dtype=self.dtype)
        
        if self.output_unpadded_inputs:
            return x,padding,freq,y,x_unpadded
        else:
            return x,padding,freq,y
    

def get_data_and_generate_train_val_test_sets(
    data_array: np.ndarray, # data matrix of shape (num_bldg,num_time_points,num_features). NOTE that features 2 and 3 are categorical features to embed time indices
    split_ratios: Union[List,Tuple], # 3-element list containing the ratios of train-val-test
    dataset_kwargs: Tuple # ONLY include num_bldg, lookback, lookahead, normalize, dtype, context_len, and output_unpadded_inputs keys. See NRELComstock definition above for details.
):
    
    assert len(split_ratios) == 3, "The split list must contain three elements."
    assert all(isinstance(i, (int, float)) for i in split_ratios), "List contains non-numeric elements"
    assert sum(split_ratios) <= 1, "Ratios must not sum upto more than 1."
    
    cum_splits = np.cumsum(split_ratios)
    
    train = data_array[:,:int(cum_splits[0]*data_array.shape[1]),:]
    val = data_array[:,int(cum_splits[0]*data_array.shape[1]):int(cum_splits[1]*data_array.shape[1]),:]
    test = data_array[:,int(cum_splits[1]*data_array.shape[1]):,:]
    
    if 0 in train.shape:
        train_set = None
        raise ValueError("Train set is empty. Possibly empty data matrix or 0 ratio for train set has been input.")
    else:
        train_set = NRELComstock(
            data_array = train,
            **dataset_kwargs
        )
        mean, std = train_set.mean, train_set.std
    if 0 in val.shape:
        val_set = None
    else:
        val_set = NRELComstock(
            data_array = val,
            mean = mean,
            std = std,
            **dataset_kwargs
        )
    if 0 in test.shape:
        test_set = None
    else:
        test_set = NRELComstock(
            data_array = test,
            mean = mean,
            std = std,
            **dataset_kwargs
        )
    
    # return the three datasets, as well as the statistics
    return train_set, val_set, test_set, mean, std