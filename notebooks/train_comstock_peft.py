import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# load timesfm stuff
import timesfm.pytorch_patched_decoder_wrapper as TFMmodel
import timesfm.custom_data_set as TFMdset
import timesfm.lora_adapter as TFMlora

# matplotlib configs
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

device, device_2 = torch.device('cuda:0'), torch.device('cuda:1') if torch.cuda.device_count() > 1 else  torch.device('cuda:0')

# batch size
batch_size = 512

# lookback and lookahead
lookback, lookahead = 96, 96

# number of buildings
num_bldg = 12

# LoRA rank
lora_rank = 1

# data
model = TFMmodel.TimesFmAppfl(lookback = lookback, lookahead = lookahead)
model.load_state_dict(torch.load('/home/sbose/timesfm/timesfm-1.0-200m-pytorch/torch_model.ckpt',map_location='cpu'))
model_lora = TFMlora.add_lora_adapters(model,lora_rank,'timesfm.stacked_transformer')

# get a dataset
data = np.load('/home/sbose/out_there/FederatedPersonalizedLoadForecasting/NRELNYdataset.npz')['data']
dataset_train, dataset_val, dataset_test, mean, std = TFMdset.get_data_and_generate_train_val_test_sets(
    data_array = data,
    split_ratios = [0.8,0.1,0.1],
    dataset_kwargs = {
        'num_bldg' : num_bldg,
        'lookback' : lookback,
        'lookahead' : lookahead,
        'normalize' : True,
        'dtype' : torch.float32,
        'context_len' : 512
    }
)

# get dataloader
dl = DataLoader(dataset_train, batch_size = batch_size, shuffle=False)
dl_test = DataLoader(dataset_test, batch_size = batch_size, shuffle=False)

# move model to device
model_lora = model_lora.to(device)

# main
if __name__ == "__main__":

    model_lora = model_lora.to(device)
    optim = torch.optim.Adam(model_lora.parameters(),lr=1e-6)
    sched = torch.optim.lr_scheduler.MultiplicativeLR(optim,lambda epoch: 1.) # Scheduler: decay the learning rate by 90% on each epoch
    steps, epoch = 0, 0
    save_every = 100
    epochs = 20
    train_len, test_len = len(dl), len(dl_test)

    for epoch in range(epochs):
        # train
        model_lora = model_lora.to(device) # move to primary cuda
        for itm in (t:=tqdm(dl)):
            start = time.time()
            x, y = itm
            x, y = x.to(device), y.to(device)
            out = model_lora(x)
            loss = F.mse_loss(out,y,reduction='mean')
            optim.zero_grad()
            loss.backward()
            optim.step()
            mse_loss, mae_loss = loss.item(), F.l1_loss(out,y,reduction='mean').item()
            if (steps == 0) and (epoch == 0):
                losses_train = pd.DataFrame({'mse_loss':[mse_loss],'mae_loss':[mae_loss]})
                losses_train.index.name = 'step'
            else:
                losses_train.loc[epoch*train_len+steps] = {'mse_loss': mse_loss, 'mae_loss': mae_loss}
            steps += 1
            if (steps % save_every == 0) or (steps==train_len):
                losses_train.to_csv('train.csv')
            end = time.time()
            print(f"Step {steps+1} on epoch {epoch+1} took {(end-start):.3f}s. MSE loss: {mse_loss:.5f}, MAE loss: {mae_loss:.5f}. Expected time for this epoch: {int((end-start)*train_len)}s")
        # step the learning rate scheduler
        sched.step()
        # test
        model_lora = model_lora.to(device_2) # move to secondary cuda
        test_mse, test_mae = 0. , 0.
        for itm in dl_test:
            x, y = itm
            x, y = x.to(device_2), y.to(device_2)
            out = model_lora(x)
            test_mse += F.mse_loss(out,y,reduction='mean').item()
            test_mae += F.l1_loss(out,y,reduction='mean').item()
        if epoch == 0:
            losses_test = pd.DataFrame({'mse_loss':[test_mse/test_len],'mae_loss':[test_mae/test_len]})
            losses_test.index.name = 'epoch'
        else:
            losses_test.loc[epoch] = {'mse_loss':test_mse/test_len,'mae_loss':test_mae/test_len}
        losses_test.to_csv('test.csv')
        torch.save(model_lora.state_dict(),'model_lora.pth')        