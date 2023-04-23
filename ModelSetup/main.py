import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from data import create_dataset
import h5py
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

fmodel = 'model_seti.pt'
fout = 'losses_seti.txt'

batchsize1 = 100
batchsize2 = 20
learning_rate = 1e-3
num_epochs = 50 #reduce this number of training accuracy is increasing while testing accuracy is decreasing. Increase if both are mediocre

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 4, 'pin_memory': True} if device=='cuda' else {}
print(f'Device used: {device}')


train_loader = create_dataset('train', seed, fin, batch_size = batchsize1, shuffle = True, **kwargs)
test_loader = create_dataset('test', seed, fin, batch_size = batchsize2, shuffle = True, **kwargs)
#valid_loader = create_dataset('valid', seed, fin, batch_size = batchsize2, shuffle = False, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,1, padding=1)
        self.conv2 = nn.Conv2d(8, 32, 3, 1, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128,1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.leakyrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.leakyrelu(self.pool(self.conv1(x)))
        out = self.leakyrelu(self.pool(self.conv2(out)))
        out = self.dropout1(out)
        out = self.flatten(out)
        out = self.leakyrelu(self.fc1(out))
        out = self.fc2(out)
        return out
    
model = Net().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

if os.path.exists(fmodel):
    print('Loading model...')
    model.load_state_dict(torch.load(fmodel))

print('Computing initial validation loss')
model.eval()
minValidLoss, points = 0., 0
for x, y in test_loader:
    with torch.no_grad():
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_NN = model(x.float())
        minValidLoss += (criterion(y_NN, y.unsqueeze(1)).item())*x.shape[0]
        points += x.shape[0]
minValidLoss /= points
print(f'Initial valid loss = {minValidLoss}')

#check if a model already exists and we should build on it
if os.path.exists(fout):
    d = np.loadtxt(fout, unpack=False)
    if d.size == 0:
        offset = 0
    else:
        offset = int(d[:,0][-1]+1) #epoch number
else: offset = 0

def train(model, device, train_loader, optimizer):
    model.train()
    loss_total = 0.0
    y_true = []
    y_pred = []
    trainLoss, points = 0., 0
    for i in train_loader:
        data, target = i
        data, target = data.to(device, non_blocking = True), target.to(device, non_blocking = True)
        
        target = target.float()
        output = model(data.float())
        loss = criterion(output, target.unsqueeze(1))
        
        loss_total += loss
        trainLoss += (loss.item())*data.shape[0]
        points+=data.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = np.round(torch.sigmoid(output.detach()).cpu().numpy())
        y_true.extend(target.tolist())
        y_pred.extend(pred.reshape(-1).tolist())
    trainLoss/=points
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy on training: {accuracy}; Train Loss: {trainLoss}')
    return accuracy, trainLoss

def test(model, device, test_loader):
    model.eval()
    y_true=[]
    y_pred=[]
    testLoss, points = 0., 0
    with torch.no_grad():
        for i in test_loader:
            data, target = i
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data.float())
            
            testLoss += (criterion(output, target.float().unsqueeze(1)).item())*data.shape[0]
            points += data.shape[0]
            
            pred = np.round(torch.sigmoid(output.detach()).cpu().numpy())
            target = target.float()
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())
        testLoss/=points
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy on testing: {accuracy}; Test Loss: {testLoss}')
    print('********************************************************')
    return accuracy, testLoss

start = time.time()
prevAccuracy = 0
for epoch in range(offset, offset+num_epochs):
    print(f'Epoch {epoch+1}/{offset+num_epochs}')
    trainAccuracy, trainLoss = train(model, device, train_loader, optimizer)
    testAccuracy, testLoss = test(model, device, test_loader)
    
    #check whether we should update the model
    if testAccuracy>prevAccuracy:
        torch.save(model.state_dict(), fmodel)
        prevAccuracy = testAccuracy
        
    f = open(fout, 'a')
    f.write(f'{epoch} {trainLoss} {testLoss}\n')
    f.close()
stop = time.time()
print(f'Time taken (min): {(stop-start)/60.0}')