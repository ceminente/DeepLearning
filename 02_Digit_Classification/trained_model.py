import numpy as np
import pandas as pd
import csv
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(3)

### Define the network class
class Net(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No, act_func, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)
        
        self.dropout = nn.Dropout(dropout)
        self.act = act_func
        self.out_softmax = nn.Softmax(dim=-1)
        self.best_config = None
        
    def forward(self, x, additional_out=False):
        
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        out = self.fc3(x)
        
        if additional_out:
            return out, torch.argmax(out,dim=-1)
        return out
    
    def train_net(self, train_data, val_data, training_pars, verbose, verbose_out):
    
        epochs = training_pars["num_epochs"]
        lr = training_pars["lr"]
        weight_decay = training_pars["weight_decay"]
        patience = training_pars["patience"]
        bs = training_pars["batchsize"]
    
        ###Define loss function
        loss_fn = nn.CrossEntropyLoss()
        ### Define an optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)
        
        ###Logs
        train_loss_log = []
        val_loss_log = []
        val_accuracy_log = []
        
        ###Dividing training data in batches and changing data format
        x_val = torch.Tensor(val_data[:,:-1])
        y_val = torch.Tensor(val_data[:,-1]).long().squeeze()
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=bs)
        
        waiting = 0
        val_loss_best = 10**4
        for num_epoch in range(epochs):#, desc="Epoch", leave=False):
            # Training
            self.train() 
            loss_batch=[]  
            for data in data_loader:
                optimizer.zero_grad()
                #dividing data and changing y
                x_train = data[:,:-1]
                y_train = data[:,-1].long().squeeze()
                #forward
                out = self.forward(x_train)
                #loss
                loss = loss_fn(out, y_train)
                loss_batch.append(loss.detach().numpy())
                #backward prop and weight update according to optimizer
                loss.backward()
                optimizer.step()
            #loss of the epoch is the mean loss over the batches
            loss = np.array(loss_batch).mean()
                
            #Validate network
            self.eval()
            with torch.no_grad(): 
                out, predicted_label = self.forward(x_val, additional_out=True)
                val_loss = loss_fn(out, y_val)
                
            if verbose:
                print('Epoch %d - Train loss: %.5f - Val loss: %.5f' % (num_epoch, float(loss), float(val_loss.data)))
           
            if val_loss <= val_loss_best:
                self.best_config = self.state_dict()
                val_loss_best = val_loss
                waiting = 0
            else:
                waiting +=1
             
            #Early stopping
            if waiting >= patience:
                if verbose:
                    print("Val loss has not improved for %d epochs ---> early stopping" %(patience))
                    print("Best validation error was at epoch %d " %(num_epoch - patience))
                if verbose_out:
                    return train_loss_log, val_loss_log, val_accuracy_log
                break
            
            if verbose_out:
                # Log
                train_loss_log.append(loss)
                val_loss_log.append(val_loss.numpy())
                val_accuracy_log.append(np.array(predicted_label.data == y_val.data).mean())
                
        if verbose_out:
            return train_loss_log, val_loss_log, val_accuracy_log
        
    def evaluate_loss(self, test_data):
        loss_fn = nn.CrossEntropyLoss()
        x_test = torch.Tensor(test_data[:,:-1])
        y_test = torch.Tensor(test_data[:,-1]).long().squeeze()
        self.eval()
        with torch.no_grad(): 
            out, predicted_label = self.forward(x_test, additional_out=True)
            test_loss = loss_fn(out, y_test)
        labels = predicted_label.numpy() == y_test.numpy()
        return test_loss.numpy(), labels.mean()


mnist = loadmat('MNIST.mat')
img = mnist["input_images"]
labels = mnist["output_labels"]

net = torch.load("Final_model.torch")
whole_dataset=np.hstack((img,labels))
test_loss, test_accuracy = net.evaluate_loss(whole_dataset)
print("After testing the whole MNIST.mat dataset the model accuracy is: ", test_accuracy.round(3))

