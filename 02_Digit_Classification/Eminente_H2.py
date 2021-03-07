#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:49:23 2021

@author: clara
"""
#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.optim as optim
import time

from sklearn.model_selection import train_test_split
from NN_H2 import Net, kfold
from tqdm import tqdm

# Set random seed
np.random.seed(3)

#%% Import dataset
from scipy.io import loadmat
mnist = loadmat('/home/clara/Documents/CNS/Homework2/MNIST.mat')
print(type(mnist))
print(mnist.keys())

img = mnist["input_images"]
labels = mnist["output_labels"]
print(img.shape)
print(img[0].shape) #28x28=784

#!WARNING: squared image must be transposed in order to be visualized correctly. This doesn't change
#much during the learning process since feed forward fully connected networks do not take into account spatial
#information

sample=5
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(np.reshape(img[sample],(28,28)).transpose(), cmap='Greys')
ax.set_title("True label: %d" %(labels[sample]), fontsize=17)
fig.tight_layout()



#%% FIRST TRAINING
#Split dataset in train and test (80% vs 20%) and training set in batches
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2, random_state=42)
train_data = np.hstack((x_train,y_train))
test_data = np.hstack((x_test,y_test))

### Initialize the network
Ni = 784
Nh1 = 200
Nh2 = 100
No = 10
func = nn.LeakyReLU()
dropout = 0.5

net = Net(Ni, Nh1, Nh2, No, func, dropout)
train_dict = {"num_epochs":100, "lr":0.0005, "weight_decay":0.0001, "patience":15, 
                  "batchsize":2000}
train_loss_log, test_loss_log, test_accuracy_log = net.train_net(train_data,test_data,train_dict,True,True)
#net.train_net(train_data,test_data,train_dict,True,False)
net.load_state_dict(net.best_config)
test_loss, test_accuracy = net.evaluate_loss(test_data)

print("Test loss: %f Accuracy: %f" %(test_loss.mean().round(3), test_accuracy.round(3)))

plt.figure(figsize=(7,5))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(test_loss_log, label='Test loss')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.grid()
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()  
 

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(test_accuracy_log, label="Accuracy")
ax.set_xlabel('Epoch', fontsize=15)
ax.set_ylabel('Accuracy', fontsize=15)
ax.grid()
ax.legend(fontsize=15)
ax.set_title("Final accuracy "+str(test_accuracy_log[-1].round(3)), fontsize=17)
fig.tight_layout()
plt.show()      

#%% MODEL SELECTION

#Prepare data and folds
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2, random_state=70)
test_data = np.hstack((x_test,y_test))
train_val_data = np.hstack((x_train,y_train))
fold_indeces = kfold(train_val_data,4)


#Prepare parameters output file
filename = "Models_2.csv"
parameters=["Ni", "Nh1", "Nh2", "No", "act_func","dropout", "num_epochs","batchsize","lr","weight_decay",
            "patience", "nfolds", "val_loss"]
list_of_models = pd.DataFrame(columns=parameters)
list_of_models.to_csv(filename,index=False)

n_models = 70
start = time.time()
with open(filename,'a',newline='') as f:
    writer=csv.writer(f)
    
    #--------FIXED PARAMETERS-------#
    Ni=784
    No=10
    n_folds = 3
    n_epochs = 150
    act_f = nn.LeakyReLU()
    patience = 15
    
    
    for nm in tqdm(range(n_models)):#, desc="Models", leave=True):
    
        #--------RANDOM SEARCH---------#
        Nh1 = np.random.randint(100,600)
        Nh2 = np.random.randint(100, Nh1)
        batchsize = np.random.choice([200,500,1000,1500,2000])
        dropout = np.random.random()*(0.5) #uniform between 0 and 0.5
        lr = 5*10**np.random.uniform(-4,-2)
        weight = 5*10**np.random.uniform(-5,-3)
        
        
        net_dict = {"Ni":Ni, "Nh1":Nh1, "Nh2":Nh2, "No":No, "act_func":act_f, "dropout":dropout}
        train_dict = {"num_epochs": n_epochs, "batchsize":int(batchsize), "lr":lr, 
                             "weight_decay":weight, "patience":patience, "nfolds": n_folds}
      
        #------TRAIN THE MODEL OVER FOLDS-----#

        CVlosses=[]
        for i in tqdm(range(len(fold_indeces))):#, desc="Folds", leave=False):
            
            ###divide validation and training set
            val_set = train_val_data[list(fold_indeces[i])]
            train_set = np.delete(train_val_data, fold_indeces[i],0)
            
            net = Net(**net_dict)
            net.train_net(train_set,val_set,train_dict,False,False)
            net.load_state_dict(net.best_config)
            val_loss, _ = net.evaluate_loss(val_set)
            CVlosses.append(val_loss)
                
        CVloss = np.array(CVlosses).mean()
        model_dict={**net_dict,**train_dict, "val_loss":CVloss}
        writer.writerow(list(model_dict.values()))
        
f.close()  

end = time.time()
print("Elapsed time to train %d models: %f s" %(n_models, end-start))
#%% TRAIN FINAL MODEL
#loading evaluated models and choosing the winning one
filename="Models.csv"
models = pd.read_csv(filename, sep=",")
best_set = models.iloc[[models.val_loss.argmin()]]
print("Set of best performing hyperparameters:")
pd.set_option('display.expand_frame_repr', False)
best_set["act_func"]=nn.LeakyReLU() #otherwise it is read as a string for some reason
print(best_set)
#save best set
best_set.to_csv("best_set.csv", index=False)

#convert best model into dictionary
best_set=best_set.drop(["val_loss"], axis=1).to_dict("r")[0]
net_pars = { your_key: best_set[your_key] for your_key in ["Ni","Nh1", "Nh2", "No", "act_func", "dropout"]}
train_pars = { your_key: best_set[your_key] for your_key in ["num_epochs", "batchsize","lr", 
                             "weight_decay","patience", "nfolds"]}#Training_pars.keys()}

#re-loading training and test set
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2, random_state=70)
test_data = np.hstack((x_test,y_test))
train_val_data = np.hstack((x_train,y_train))
#shuffling training data and dividing it in validation and train 
#(validation is used for early stopping and monitoring performances)
np.random.shuffle(train_val_data)
train_set = train_val_data[:int(np.ceil(8*len(train_val_data)/10))]
val_set = train_val_data[int(np.ceil(8*len(train_val_data)/10)):]

#final training and saving the final model
net = Net(**net_pars)
train_loss_log, val_loss_log, val_accuracy_log=net.train_net(train_set,val_set,train_pars,True,True)
torch.save(net, "Final_model.torch")

#%% plotting final training
plt.figure(figsize=(7,5))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(val_loss_log, label='Test loss')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.grid()
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()  
 
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(val_accuracy_log, label="Accuracy")
ax.set_xlabel('Epoch', fontsize=15)
ax.set_ylabel('Accuracy', fontsize=15)
ax.grid()
ax.legend(fontsize=15)
ax.set_title("Final accuracy on validation data "+str(val_accuracy_log[-1].round(3)), fontsize=17)
fig.tight_layout()
plt.show() 

#%% printing best set
best_set = pd.read_csv("best_set.csv")
print("Set of best performing hyperparameters:")
pd.set_option('display.expand_frame_repr', False)
best_set["act_func"]=nn.LeakyReLU() #otherwise it is read as a string for some reason
print(best_set)

#%% testing final model on test set and whole MNIST.mat
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2, random_state=70)
test_data = np.hstack((x_test,y_test))

net = torch.load("Final_model.torch")
test_loss, test_accuracy = net.evaluate_loss(test_data)

print("After testing on unseen data (test set) the model accuracy is: ", test_accuracy.round(3))
print("Corresponding to a CrossEntropyLoss of:", test_loss.round(3))
whole_dataset=np.hstack((img,labels))
test_loss, test_accuracy = net.evaluate_loss(whole_dataset)
print("After testing on whole dataset the model accuracy is: ", test_accuracy.round(3))
print("Corresponding to a CrossEntropyLoss of:", test_loss.round(3)) 

#%% OUPUT AND CORRESPONDING IMAGES
with torch.no_grad(): 
    out, predicted_label = net.forward(torch.Tensor(x_test[12:20,]), additional_out=True)
fig, ax = plt.subplots(8,2, figsize=(7,15))
i=0
for prob,label in zip(out.numpy(), predicted_label.numpy()):
    ax[i][0].imshow(x_test[12+i].reshape(28,28).transpose())
    ax[i][0].axis("off")
    ax[i][1].bar(np.arange(0,10), softmax(prob))
    ax[i][1].set_xticks(np.arange(0,10))
    ax[i][1].set_title("True label:"+str(y_test[12+i].squeeze().astype(int)), fontsize=14)
    i+=1
fig.tight_layout()
fig.savefig("Out_visualization.png")
plt.show()

# %% WEIGHTS
fc1_weights = net.fc1.weight.data.t()
fc2_weights = net.fc2.weight.data.t()
fc3_weights = net.fc3.weight.data.t()
fig,ax = plt.subplots(2,2, figsize=(9,9))
#LAYER 1
n=0
for i in range(2):
    for j in range(2):
        im = ax[i][j].imshow(fc1_weights[:,n].reshape(28,28), cmap="Greys")
        ax[i][j].axis("off")
        n+=1
#fig.colorbar(im, ax=ax.ravel().tolist())
fig.tight_layout()
fig.savefig("recFieldL1.png")
plt.show()
fig,ax = plt.subplots(3,3, figsize=(12,12))
#OUTPUT LAYER
n=0
for i in range(3):
    for j in range(3):
        im=ax[i][j].imshow(torch.matmul( torch.matmul(fc1_weights, fc2_weights),
                                   fc3_weights[:,n] ).reshape(28,28).t(), cmap="Greys")
        ax[i][j].axis("off")
        ax[i][j].set_title("Neuron: "+str(n), fontsize=18)
        n+=1
#fig.colorbar(im, ax=ax.ravel().tolist())
fig.tight_layout()
fig.savefig("recFieldLout.png")
plt.show()

#%% GRADIENT ASCENT FOR FEATURE VISUALIZATION
def Ascent_img(layer, neuron, lr, epochs):
    img = torch.rand(784, requires_grad=True) 
    optimizer = torch.optim.Adam([img], lr=lr, weight_decay=5e-5)
    for num_ep in range(epochs):  
        optimizer.zero_grad()
        activated = -from_img_to_activation(net,layer,img)[neuron] 
        activated.backward()
        optimizer.step()
    return img

def from_img_to_activation(net, layer, img):
    if layer == 1:
        return net.network.act(net.network.fc1(img))
    if layer == 2:
        return net.act(net.fc2(net.act(net.fc1(img))))
    if layer == 'out':
        return net.fc3(net.act(net.fc2(net.act(net.fc1(img)))))
    
l = 'out'   
fig,ax = plt.subplots(3,3, figsize=(9,9))
n=0
for i in range(3):
    for j in range(3):
        img = Ascent_img('out', n, 0.0001, 10000) 
        ax[i][j].imshow(img.data.reshape(28,28).t(), cmap="Greys")
        ax[i][j].set_title("Neuron: "+str(n), fontsize=18)
        ax[i][j].axis("off")
        n+=1
fig.tight_layout()
fig.savefig("Ascent.png")
plt.show()


# %%
