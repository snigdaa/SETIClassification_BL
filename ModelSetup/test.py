import numpy as np
import torch
import sys,os,h5py
from data import create_dataset
import matplotlib.pyplot as plt
from main import Net

fin = '/path/to/testfile.h5'

seed = 123
testbs = 300 #reduce this if you're running out of memory

fout = 'losses_seti.txt'
fmodel = 'model_seti.pt'

device = device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 4, 'pin_memory': True} if device=='cuda' else {}
print(f'Device used: {device}')

fin = h5py.File(fin)['target'] #change key if you've named the array of inputs something else
testInputs = torch.from_numpy(np.load(fin)).to(device) #change this if your structure is not just a npy file
                                
model = Net()
model.load_state_dict(torch.load(fmodel, map_location=torch.device(device))).to(device)

model.eval()

with torch.no_grad():
    pred = model(testInputs)

pred = pred.detach().cpu().numpy()
np.save(pred, 'modelPredictedAnswers.npy')

        