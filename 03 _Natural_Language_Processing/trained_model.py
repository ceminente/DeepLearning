import argparse
import torch
import numpy as np
import re
import string
import gensim
import time
import itertools
import pickle
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import KFold
import pandas as pd
import csv
from scipy.special import softmax
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")


##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Generate the rest of the sentence given a seed')

# Seed
parser.add_argument('--seed',    type=str,   default='It was a dark and stormy night', help='Initial part of the sentence')
#Length
parser.add_argument('--length',     type=int,   default=1,    help='Number of words to generate')

##############################


##############################
## NETWORK CLASS
##############################
class Network(nn.Module):
    def __init__(self, len_vocab, emb_dim, hidden_units, layers, dropout):
        super().__init__()

        #Creating pre-trained embedding layer at the beginning using the weight already computed
        embedding_matrix= torch.load(path+"embedding_model/embedding.torch")
        embedding_matrix = torch.FloatTensor(embedding_matrix)
        self.embedding = nn.Embedding(len_vocab, emb_dim).from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad=False   #pretrained

        self.rnn = nn.LSTM(input_size=emb_dim, 
                           hidden_size=hidden_units,
                           num_layers=layers,
                           dropout=dropout, 
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(hidden_units, len_vocab)
        
    def forward(self, x, state=None):
        #Embedding
        x = self.embedding(x)
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        x = F.leaky_relu(self.out(x))

        return x, rnn_state

    
    def train_epoch(self, dataloader, loss_fn, optimizer):
        # Training
        self.train()
        train_loss = []
        for batch_sample in dataloader:
            sample_batch = batch_sample['encoded']
            x = sample_batch[:, :-1].to(device) #all words besides the last
            y = sample_batch[:, 1:].to(device)  #all words besides the first one
            optimizer.zero_grad()
            predicted, _ = self.forward(x) #predicted has size (batchsize, 24, len_vocab)
            loss = loss_fn(predicted.transpose(1, 2), y) #transpose so that predicted and y match
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data.cpu().numpy())
        return np.mean(np.array(train_loss))



    def test_epoch(self, test_dataloader, loss_fn):
        # Validation
        self.eval() # Evaluation mode (e.g. disable dropout)
        test_loss = []
        with torch.no_grad(): # No need to track the gradients
            for batch_sample in test_dataloader:
                sample_batch = batch_sample['encoded']
                x = sample_batch[:, :-1].to(device) #all words besides the last
                y = sample_batch[:, 1:].to(device)  #all words besides the first one
                predicted, _ = self.forward(x) #predicted has size (batchsize, 24, len_vocab)
                loss = loss_fn(predicted.transpose(1, 2), y) 
                test_loss.append(loss.data.cpu().numpy())
        return np.mean(np.array(test_loss))


    def train_full(self, epochs, patience, train_dataloader, val_dataloader, loss_fn, optimizer, verbose, filename):

        val_loss_best = float(1000)
        train_loss_log=[]
        val_loss_log=[]
        for epoch in range(epochs):

            start = time.time()

            train_loss = self.train_epoch(train_dataloader, loss_fn=loss_fn, optimizer=optim) 
            val_loss = self.test_epoch(val_dataloader, loss_fn=loss_fn) 

            end = time.time()

            #Print Validation loss

            if verbose:
                print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, epochs, val_loss))
                print("\n Time elapsed for one epoch:", end-start)

            if val_loss <= val_loss_best:
                # Save network parameters
                torch.save(self, path+filename)
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
            torch.cuda.empty_cache()
            
        return train_loss_log, val_loss_log


    def predict(self, seed_text, text_length):

        clean_seed_text = preprocess(seed_text, load=False, filename="temp.txt", is_seed=True)
        clean_seed_text = clean_seed_text.split()
        w2i = pickle.load(open(path+"embedding_model/w2i.p", "rb"))  
        i2w = pickle.load(open(path+"embedding_model/i2w.p", "rb")) 
        w2s = pickle.load(open(path+"embedding_model/w2s.p", "rb"))
        encoded = encode(w2i, clean_seed_text)
        encoded = encoded.unsqueeze(0).to(device)
        print("\n\n"+seed_text, end='', flush=True)

        #Creating context (needed to create first hidden state)
        self.eval() 
        with torch.no_grad():
            out, hstate = self(encoded)
            next = self.softmax_sampling(out[:, -1, :].cpu().numpy())
            decoded = i2w[next]
            print(w2s[decoded], end=' ', flush=True)
            encoded = torch.LongTensor([next])
            encoded = encoded.unsqueeze(0).to(device)
        for words in range(text_length-1):
            with torch.no_grad():
                out, hstate = self(encoded, hstate)
                next = self.softmax_sampling(out[:, -1, :].cpu().numpy())
                decoded = i2w[next]
                print(w2s[decoded], end=' ', flush=True)
                encoded = torch.LongTensor([next])
                encoded = encoded.unsqueeze(0).to(device)
        print("\n\n")

    def softmax_sampling(self,x,return_prob=False):
        prob = softmax(x)
        vocab_idx = np.arange(prob.shape[1])
        prob = prob.reshape(-1,) #remove batch dim
        next_word = np.random.choice(vocab_idx, p=prob)
        if return_prob:
            return next_word.item(), prob
        return next_word.item()

##############################


##############################
## PREPROCESSING FUNCTION
##############################
def preprocess(input, load, filename='/text/austen.txt', is_seed=False):

    if load:

        austen = open(path+filename, 'r').read()
        print("Opened "+ str(filename))
        alphabet = list(set(austen))
        alphabet.sort()
        #print('Found letters:', alphabet)
        return austen

    else:

        if is_seed:
            austen = input
        else:
            titles = input
            #building one single text
            texts_list=[open(path+"/text/"+text, 'r').read() for text in titles]
            austen = ""
            for text in texts_list:
                austen+=text

        #removing chapter headers
        austen = re.sub(r'CHAPTER.*\n+', '\n\n', austen)
        austen = re.sub(r'Chapter.*\n+', '\n\n', austen)

        #lowercase
        austen = austen.lower()

        #remove newlines inside text (no "true" newlines)
        austen = re.sub('(?<=[^.])\n'," ",austen)
        
        #sobstituting these characters with spaces
        odd_characters = ['"', '&', "'", '(', ')', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '_', '£', '—', '‘', '’', '“', '”']
        rx = '[' + re.escape(''.join(odd_characters)) + ']'
        austen = re.sub(rx, ' ', austen)

        #sobstituting some punctuation with "."
        punct_to_point = ['!', ':', ';']
        rx = '[' + re.escape(''.join(punct_to_point)) + ']'
        austen = re.sub(rx, '.', austen)

        #sobstituting accented vowels with "normal" version
        accented = ['à', 'é', 'ê']
        not_accented = ['a', 'e', 'e']
        for a, na in zip(accented, not_accented):
            austen = re.sub(a, na, austen)

        #sobstituting residual punctuation with words
        austen = re.sub('['+','+']', ' '+'commapunct'+' ', austen)
        austen = re.sub('['+'.'+']', ' '+'fullstoppunct'+' ', austen)
        austen = re.sub('['+'?'+']', ' '+'questpunct'+' ', austen)

        #remove multiple spaces
        austen = re.sub(' +', ' ', austen)

        alphabet = list(set(austen))
        alphabet.sort()
        #print('Found letters:', alphabet)

        import codecs
        with codecs.open(path+filename, 'w', encoding="UTF-8") as F:
            F.write(austen)

        return austen

def encode(dict_w2i, text):
    encoded = [dict_w2i[word] for word in text]
    return torch.LongTensor(encoded)

def decode(dict_i2w, numeric_text):
    decoded = [dict_i2w[num] for num in numeric_text]
    return decoded


##############################

if __name__ == '__main__':

    #Set path
    usingColab=False
    if usingColab:
        from google.colab import drive
        drive.mount('/content/drive')
        path = '/content/drive/My Drive/HW5_DeepLearning/'
    else:
        path = ''

    
    # Parse input arguments
    args = parser.parse_args()

    net = torch.load(path+"Final_model.torch", map_location=torch.device('cpu'))
    net.predict(args.seed, args.length)


