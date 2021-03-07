import os
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import csv
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from time import time
from sklearn.model_selection import KFold
from sklearn import manifold

from scipy.io import loadmat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# GPU
usingCuda = False
if torch.cuda.is_available():
  usingCuda = True
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
print('Using Cuda:', usingCuda)

usingColab=False
if usingColab:
    from google.colab import drive
    drive.mount('/content/drive')
    path = '/content/drive/My Drive/DeepLearning/HW4/'
else:
    path=''


### Autoencoder class

class Autoencoder(nn.Module):
    
    def __init__(self, encoded_space_dim, dropout):
        super().__init__()

        self.encoded_space_dim=encoded_space_dim
        self.dropout=dropout

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout(self.dropout)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.encoded_space_dim)
        )

        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(self.encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True),
            nn.Dropout(self.dropout)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x

    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    def train_epoch(self, dataloader, loss_fn, optimizer):
        # Training
        self.train()
        loss_log = []
        for train_batch in dataloader:
            image_batch = train_batch[0].to(device) 
            #Forward
            output = self(image_batch)
            loss = loss_fn(output, image_batch)
            # Backward 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.data.cpu().numpy())
        loss_log = np.asarray(loss_log)
        return np.mean(loss_log)

    def train_epoch_mod(self, dataloader, loss_fn, optimizer):
        # Training
        self.train()
        loss_log = []
        for train_batch in dataloader:
            mod_batch = train_batch[1].to(device)
            image_batch = train_batch[0].to(device)
            #Forward
            output = self(mod_batch)
            loss = loss_fn(output, image_batch)
            # Backward 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.data.cpu().numpy())
        loss_log = np.asarray(loss_log)
        return np.mean(loss_log)

    def test_epoch(self, dataloader, loss_fn, additional_out=False):
        # Validation
        self.eval() # Evaluation mode (e.g. disable dropout)
        with torch.no_grad(): # No need to track the gradients
            conc_out = torch.Tensor().float()
            conc_label = torch.Tensor().float()
            for sample_batch in dataloader:
                # Extract data and move tensors to the selected device
                image_batch = sample_batch[0].to(device)
                # Forward pass
                out = net(image_batch)
                # Concatenate with previous outputs
                conc_out = torch.cat([conc_out, out])
                conc_label = torch.cat([conc_label, image_batch]) 
            # Evaluate global loss
            val_loss = loss_fn(conc_out, conc_label)
            if additional_out:
                return val_loss.data.cpu().numpy(), conc_out.cpu().numpy(), conc_label.cpu().numpy()
            else:
                return val_loss.data.cpu().numpy()

    def test_epoch_mod(self, dataloader, loss_fn, additional_out=False):
        # Validation
        self.eval() # Evaluation mode (e.g. disable dropout)
        with torch.no_grad(): # No need to track the gradients
            conc_out = torch.Tensor().float()
            conc_label = torch.Tensor().float()
            for sample_batch in dataloader:
                # Extract data and move tensors to the selected device
                mod_batch = sample_batch[1].to(device)
                image_batch = sample_batch[0].to(device)  #! train_batch[0]=img x 512, train_batch[1]=noised_img x 512, train_batch[2]=label x 512
                # Forward pass
                out = net(mod_batch)
                # Concatenate with previous outputs
                conc_out = torch.cat([conc_out, out])
                conc_label = torch.cat([conc_label, image_batch]) 
            # Evaluate global loss
            val_loss = loss_fn(conc_out, conc_label)
            if additional_out:
                return val_loss.data.cpu().numpy(), conc_out.cpu().numpy(), conc_label.cpu().numpy()
            else:
                return val_loss.data.cpu().numpy()

            
    def train_full(self, epochs, patience, train_dataloader, val_dataloader, loss_fn, optimizer, verbose, denoise=False):
        val_loss_best = float(1000)
        train_loss_log=[]
        val_loss_log=[]
        for epoch in range(epochs):

            start=time()

            if denoise:
                train_loss = self.train_epoch_mod(dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim) 
                val_loss = self.test_epoch_mod(dataloader=val_dataloader, loss_fn=loss_fn) 
            else:
                train_loss = self.train_epoch(dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim) 
                val_loss = self.test_epoch(dataloader=val_dataloader, loss_fn=loss_fn) 

            end = time()

            #Print Validationloss

            if verbose:
                print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, epochs, val_loss))
                print("\n Time elapsed for one epoch:", end-start)

            if val_loss <= val_loss_best:
                # Save network parameters
                torch.save(self, path+'denoise_model_esd_'+str(int(encoded_space_dim))+'.torch')
                val_loss_best = val_loss
                waiting = 0
            else:
                waiting +=1

            #Early stopping
            if waiting >= patience and epoch > 20:
                return train_loss_log, val_loss_log
                print("Val loss has not improved for %d epochs ---> early stopping" %(patience))
                print("Best validation error was at epoch %d " %(epoch - patience))
                break

            train_loss_log.append(train_loss)
            val_loss_log.append(val_loss)
            
        return train_loss_log, val_loss_log


    def get_encoded(self, dataset, label_present=False):

        encoded_imgs = []
        labels = []

        for sample in tqdm(dataset):
            if label_present:
                img = sample[0].unsqueeze(0).to(device)
                label = sample[1]
            else: 
                img = sample.unsqueeze(0).to(device)

            # Encode image
            self.eval()
            with torch.no_grad():
                encoded_img  = self.encode(img)

            encoded_imgs.append(encoded_img.flatten().cpu().numpy())
            if label_present:
                labels.append(label)

        if label_present:
            return encoded_imgs, labels
        else: return encoded_imgs

    def get_decoded(self, encoded_sample, return_img=False, **kwargs):
        filename = kwargs.get('filename', None)
        self.eval()
        with torch.no_grad():
            encoded_torch = torch.tensor(encoded_sample).float().unsqueeze(0)
            img  = self.decode(encoded_torch)

        if return_img:
            plt.figure(figsize=(8,6))
            plt.imshow(img.squeeze().cpu().numpy(), cmap='gist_gray')
            plt.savefig(path+"img/"+filename)
            plt.show()
            plt.close()
        else:
            return img

    
    def encoded_space_walk(self, dataset, start, end, filename, steps=20):
        
        #prepare dataset for masking (done via numpy and separating img and labels)
        test_loader = DataLoader(dataset, batch_size=len(dataset),  shuffle=False)
        test_dataset_array = next(iter(test_loader))[0].numpy()
        test_dataset_labels = next(iter(test_loader))[1].numpy()

        centroids=[]

        l = [start,end]
        #select only imgs with given label, encode them and compute centroids for that label
        for label in l:
            mask = test_dataset_labels==label
            dataset_l = torch.tensor(test_dataset_array[mask])
            encoded_dataset_l = net.get_encoded(dataset_l, label_present=False)
            centroids.append(np.array(encoded_dataset_l).mean(axis=0))

        trajectory=[]
        #create trajectory from one centroid to another
        for dim in range(self.encoded_space_dim):
            trajectory.append( np.linspace(centroids[0][dim], centroids[1][dim], steps) )

        #each row is an encoded representation, we have as many rows as the steps we are taking from start to end
        trajectory_samples=np.array(trajectory).transpose()#.reshape((-1,self.encoded_space_dim))

        #decode and plot
        cols=int(steps/2)
        fig, axs = plt.subplots(2, cols, figsize=(15,5))
        i=0
        for sample,ax in zip(trajectory_samples, axs.flatten()):
            #decode
            img = net.get_decoded(sample)
            #plot
            ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
            ax.set_title('step: %d' %i)
            ax.set_xticks([])
            ax.set_yticks([])
            i+=1
        fig.tight_layout()
        fig.savefig(path+"img/"+filename)
        plt.show()

#load dataset and transform it to torch tensor
mnist = loadmat(path+'MNIST.mat')
img = mnist["input_images"].reshape((-1,28,28)) #reshape
dataset = torch.from_numpy(img).unsqueeze(1) #add channel dimension
#load net in evaluation mode    
net = torch.load("Final_model.torch")  
net.to(device)
net.eval() 
loss_fn = torch.nn.MSELoss()
with torch.no_grad():
    out = net.forward(dataset) 
loss = loss_fn(out, dataset)

print("\n\n\tAfter feeding the MNIST.mat dataset to the autoencoder the reconstrction error is:", round(float(loss),3))
print("\n\n")
