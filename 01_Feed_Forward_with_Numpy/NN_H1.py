import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from tqdm import tnrange

def get_XY(dataset):
    '''
    This function divides a dataset in features and labels and 
    returns them in the shape [n_samples, 1]

    INPUT
    dataset: np.array, first column must be x values, second 
            column must be y values (labels)

    OUTPUT
    X,Y: np.arrays of size [n_samples,1]
    '''
    X = dataset[:,0].reshape(-1,1)
    Y = dataset[:,1].reshape(-1,1)
    return X, Y


def kfold(dataset, k, shuffle=False):
    '''
    This function returns a list of k lists, each containing the 
    indeces of the data in each fold

    INPUT:
    dataset: np.array of data to be split
    k: integer, number of folds
    shuffle: bool, if True indeces are reshuffled before splitting
    
    OUTPUT:
    folds: np.array, each array contains the indeces of the k-th fold
    '''
    n_samples = len(dataset)
    indeces = np.arange(0,len(dataset),1).astype(int)
    if shuffle == True:
        np.random.shuffle(indeces)
    folds = list(np.array_split(indeces, k))
    return folds


def divide_in_batches(train, b_dim, shuffle=False):
    
    '''
    This function divides the "train" dataset in batches of dimension 
    b_dim.

    INPUT: 
    train: train dataset
    b_dim: integer, batch dimension
    shuffle: 
    
    OUTPUT:
    XY_lst: list of batches (X and Y already separated)
    
    #N.B. last batch is smaller if len(train)%b_dim!=0
    '''
    
    #optional shuffle
    if shuffle==True: np.random.shuffle(train)
    #print warning
    #if len(train)%b_dim!=0 and warning==True: 
     #   print('Warning: last batch has size %d instead of %d' %(len(train)%b_dim,b_dim))
    #number of batches
    n = int(np.ceil(len(train)/b_dim))
    #get batches
    b_lst = [train[i*b_dim:(i+1)*b_dim] for i in range(n)]
    #divide in X and Y
    XY_lst = [get_XY(batch) for batch in b_lst]
    
    return XY_lst



#FUNCTION TO USE IN ORDER TO DEFINE ACTIVATION FUNCTION AND ITS DERIVATIVE
def softplus(x):
    return np.log(1+np.exp(x))

def act_function(f = 'sig', leak = 0.05):
    if f == 'sig':
        act = expit 
        act_der = lambda x: act(x) * (1 - act(x))
    if f == 'ReLU':
        act = lambda x: x*(x>0)
        act_der = lambda x: (x>0).astype(int)
    if f == 'leaky ReLU':
        act = lambda x: x*(x>0) + leak*x*(x<0)
        act_der = lambda x: (x>0).astype(int) + leak*(x<0)
    if f == 'softplus':
        act = lambda x: softplus(x)
        act_der = expit
    return act, act_der

#FUNCTION TO USE IN ORDER TO DEFINE REGULARIZATION TERM (actually, its derivative)
def reg(f="None", alpha=0.001):
    if f=="None": 
        reg_term = lambda x: 0
    if f=='L1': 
        reg_term = lambda x: np.vstack((alpha*np.ones(x[:-1,:].shape), 0*x[-1, :]))
    if f=='L2': 
        reg_term = lambda x: np.vstack((alpha*x[:-1,:], 0*x[-1, :]))
    return reg_term


#%% Network class


#the matrices in the class have been transposed so now weights matrices have dimension
#(N_in, N_out)
#everything was changed in order to have (when needed, i.e. not in the wieghts matrices, for example)
#the batchsize as first dimension.


class Network():
    
    def __init__(self, Ni, Nh1, Nh2, No, act_f='sig'):
           
        ### WEIGHT INITIALIZATION (Xavier)
        # Initialize hidden weights and biases (layer 1)
        Wh1 = (np.random.rand(Ni, Nh1) - 0.5) * np.sqrt(12 / (Nh1 + Ni))
        Bh1 = np.zeros([1, Nh1])
        self.WBh1 = np.concatenate([Wh1, Bh1]) # Weight matrix including biases
        # Initialize hidden weights and biases (layer 2)
        Wh2 = (np.random.rand(Nh1, Nh2) - 0.5) * np.sqrt(12 / (Nh2 + Nh1))
        Bh2 = np.zeros([1, Nh2])
        self.WBh2 = np.concatenate([Wh2, Bh2]) # Weight matrix including biases
        # Initialize output weights and biases
        Wo = (np.random.rand(Nh2, No) - 0.5) * np.sqrt(12 / (No + Nh2))
        Bo = np.zeros([1, No])
        self.WBo = np.concatenate([Wo, Bo]) # Weight matrix including biases
        
        ### ACTIVATION FUNCTION
        self.act, self.act_der = act_function(act_f)
        
    def forward(self, X, additional_out=False):
        
        ''' 
            This function was changed just to account for 
            the change in dimension of the matrices so just the matrix products were affected
        '''
                
        ### Hidden layer 1
        # Add bias term
        #X = np.append(X, 1)
        bias = np.ones([X.shape[0],1])
        X = np.hstack([X,bias])

        # Forward pass (linear)
        H1 = np.matmul(X, self.WBh1)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        bias = np.ones([Z1.shape[0],1])
        Z1 = np.hstack([Z1,bias])
        # Forward pass (linear)
        H2 = np.matmul(Z1, self.WBh2)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        bias = np.ones([Z2.shape[0],1])
        Z2 = np.hstack([Z2,bias])
        # Forward pass (linear)
        Y = np.matmul(Z2, self.WBo)
        
        if additional_out:
            return Y, Z2
        
        return Y
            
    def update(self, X, label, lr, reg_term=(lambda x:0)):
        
        '''
        INPUT:

        X: np.array, batch of features
        label: np.array, batch of labels
        lr: float, learning rate
        reg_term: lambda function, regularization term (output of reg function)

        OUTPUT:
        loss: mean MSE over the batch 

        !the regularization term is just included in the backprop, not in the 
        output MSE
        '''
               
        ### Hidden layer 1
        # Add bias term
        #X = np.append(X, 1)
        bias = np.ones([X.shape[0],1])
        X = np.hstack([X,bias])

        # Forward pass (linear)
        H1 = np.matmul(X, self.WBh1)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        bias = np.ones([Z1.shape[0],1])
        Z1 = np.hstack([Z1,bias])
        # Forward pass (linear)
        H2 = np.matmul(Z1, self.WBh2)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        bias = np.ones([Z2.shape[0],1])
        Z2 = np.hstack([Z2,bias])
        # Forward pass (linear)
        Y = np.matmul(Z2, self.WBo)
        # NO activation function
        
        # Evaluate the derivative terms
        D1 = Y - label
        D2 = Z2
        D3 = self.WBo[:-1,:] 
        D4 = self.act_der(H2)
        D5 = Z1
        D6 = self.WBh2[:-1,:]
        D7 = self.act_der(H1)
        D8 = X
        
        # Layer Error
        Eo = D1
        Eh2 = np.matmul(Eo, D3.T) * D4
        Eh1 = np.matmul(Eh2, D6.T) * D7
        
        
        # Derivative for weight matrices
        dWBo = np.einsum("ij, ih -> ijh", D2,Eo)
        dWBh2 = np.einsum("ij, ih -> ijh", D5, Eh2)
        dWBh1 = np.einsum("ij, ih -> ijh", D8, Eh1)
        
        # Update the weights
        self.WBh1 -= lr * (np.mean(dWBh1, axis=0) + reg_term(self.WBh1))
        self.WBh2 -= lr * (np.mean(dWBh2, axis=0) + reg_term(self.WBh2))
        self.WBo -= lr * (np.mean(dWBo, axis=0) + reg_term(self.WBo))
        
        # Evaluate loss function
        loss = np.mean((Y - label)**2)
        
        return loss
    
    def plot_weights(self):
    
        fig, axs = plt.subplots(3, 1, figsize=(12,6))
        axs[0].hist(self.WBh1.flatten(), 20)
        axs[1].hist(self.WBh2.flatten(), 50)
        axs[2].hist(self.WBo.flatten(), 20)
        plt.legend()
        plt.grid()
        plt.show()


def train_model(model, train_set, val_set, batchsize,
                n_epochs, lr, decay=True, lr_final=0.0001, reg_type=None, 
                alpha_reg=0.0001, patience = 300, return_log=False):
    
    '''
    INPUT:
    
    model: a network of class Network
    train_set: training dataset, must contain both features and labels
    val_set: validation dataset, must contain both features and labels
    batchsize: integer, number of samples in each batch
    num_epochs: integer, number of epochs for training. During one epoch all the 
                batches are used for training (so all the dataset is used in one epoch)
    lr: float, learning rate
    en_decay: bool, if True learning rate is decreased so that last value is lr_final
    lr_final: last value of learning rate
    reg_type: str, regularizer type
    alpha_reg: constant of regularization
    patience: number of epochs to wait before stopping the training if validation error does not improve
    return_log: bool, if True train loss and validation loss at every iteration are returned
    
    
    OUTPUT:
    model: network of class Network, trained model that got best validation score
    best_train_loss: float, best train loss 
    best_val_loss: float best validation loss 
    train_loss_log: list of train losses (up until stop for early stopping 
                                          so that best_train_loss = train_loss_log[-patience])
    val_loss_log: list of validation losses (up until stop for early stopping 
                                          so that best_train_loss = train_loss_log[-patience])
    
    '''
    
    train_loss_log = []
    val_loss_log = []
    
    best_val_loss = 1000
    count = 0
    best_model = model
    best_train_loss = 1000

    #compute dacay given final value and number of epochs
    lr_decay = (lr_final / lr)**(1 / n_epochs) 

    #set regularization function
    reg_term = reg(reg_type, alpha_reg)

    for num_ep in tnrange(n_epochs, desc="Epochs", leave=False):
        # Learning rate decay
        if decay:
            lr *= lr_decay

        #create batches given batchsize: batches is a list of lists, each one has x and y 
        batches = divide_in_batches(train_set, batchsize) 

        #train each batch
        train_loss_vec = [model.update(x, y, lr, reg_term) for x, y in batches] 
        #compute train loss as mean over batch losses 
        #(loss of a single batch is the mean of the losses of that batch )
        train_loss = np.mean(np.array(train_loss_vec))

        #compute validation loss
        x_val, y_val = get_XY(val_set)
        y_val_est = model.forward(x_val)
        val_loss = np.mean((y_val_est - y_val)**2)
        
        # Update logs
        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        
        #early stopping
        if (val_loss <= best_val_loss):
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_model = model
            count = 0 #count is reset everytime i get a better performance
        elif val_loss > best_val_loss:
            count += 1
        if (count >= patience):
            break

    if return_log==True:
        return best_model, best_train_loss, best_val_loss, train_loss_log, val_loss_log
        #returns both last iteration and logs
    else:
        return best_model, best_train_loss, best_val_loss #only returns last iteration