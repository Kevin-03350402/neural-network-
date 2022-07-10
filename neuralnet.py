import numpy as np
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)

args = parser.parse_args()

# read in arrays and files 

xtrain = args2data(args)[0] # with bias folded in 
ytrain = args2data(args)[1] 
xtest = args2data(args)[2] # with bias folded in 
ytest = args2data(args)[3]

trainf = args2data(args)[4]
trainfile = open(trainf,"w+")

testf = args2data(args)[5]
testfile = open(testf,"w+")

metricf = args2data(args)[6]
metricfile= open(metricf,"w+")

nepochs = args2data(args)[7]

hiddenunit = args2data(args)[8]

init_flag = args2data(args)[9]

lr = args2data(args)[10]

def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random initialization here
    # figure out the rows in shape
    nrows = shape[0]
    # figure out the columns in shape
    ncols = shape[1]
    random = np.random.uniform(-0.1,0.1,size = (nrows,ncols))
    random[:,0] = 0
    return random

def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # figure out the rows in shape
    nrows = shape[0]
    # figure out the columns in shape
    ncols = shape[1]
    z = np.zeros((nrows,ncols))
    return(z)




ex1 = xtrain[0,:]
w1col = len(ex1)
ex1 = ex1.reshape(w1col,1)
w1row = hiddenunit
shapew1 = (w1row,w1col)
w1i = random_init(shapew1)

# xtrain has 500 rows ans 129 columns, each row is an example, first term bias
# W*x = signal 

def linear(X, w):
    """
    Implement linear forward
    :param X: input logits of shape (input_size)
    :param w: weights of shape (hidden_size, input_size) or (output_size, hidden_size + 1)
    :return: linear forward output of shape (hidden_size) or (output_size)
    """


    signal = np.matmul(w,X)
    return signal

sign = (linear(ex1,w1i))

def sigmoid(a):
    """
    Implement sigmoid function.
    :param a: input logits of shape (hidden_size)
    :return: sigmoid output of shape (hidden_size)
    """
    e = np.exp(a)
    return e / (1 + e)

def softmax(z):
    """
    Implement softmax function.
    :param z: input logits of shape (output_size)
    :return: softmax output of shape (output_size)
    """
    ze = np.exp(z)
    res = ze/np.sum(ze)
    return res


z = sigmoid(sign)



def cross_entropy(y, y_hat):
    """
    Compute cross entropy loss.
    :param y: label
    :param y_hat: prediction
    :return: cross entropy loss
    """
    yv = np.zeros(len(y_hat))
    yv[y] = 1
    # find the log array 
    yhl = np.log(y_hat)
    loss = -(np.dot(yv,yhl))
    return(loss)

def d_linear(X, w):
    return(w.T, X.T)

def d_sigmoid(Z):
    """
    Compute gradient of sigmoid output w.r.t. its input.
    :param Z: sigmoid's input
    :return: gradient
    """

    
    dia =Z-Z*Z
    #(1-z^2)
    # it is a matrix!

    return (dia)



def d_cross_entropy_vec(y, y_hat):
    """
    Compute gradient of loss w.r.t. ** softmax input **.
    Note that here instead of calculating the gradient w.r.t. the softmax probabilities,
    we are directly computing the gradient w.r.t. the softmax input.
    Try derive the gradient yourself, and you'll see why we want to calculate this in a single step
    :param y: label of shape (output_size)
    :param y_hat: predicted softmax probability of shape (output_size)
    :return: gradient of shape (output_size)
    """
    # want djdb
    yv = np.zeros(len(y_hat))
    yv[y] = 1
    return((y_hat - yv).reshape(len(y_hat - yv),1))

class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer *including* the folded bias
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        self.w1 = weight_init_fn([self.n_hidden, self.n_input])
        self.w2 = weight_init_fn([self.n_output,self.n_hidden+1])

        # initialize parameters for adagrad
        self.epsilon1 = (0.00001)*np.ones((self.w1.shape[0],self.w1.shape[1]))
        self.epsilon2 = (0.00001)*np.ones((self.w2.shape[0],self.w2.shape[1]))
        self.grad_sum_w1 = np.zeros((self.w1.shape[0],self.w1.shape[1]))
        self.grad_sum_w2 = np.zeros((self.w2.shape[0],self.w2.shape[1]))

        # feel free to add additional attributes
        self.x = np.zeros(self.n_input)
        self.a = linear(self.x, self.w1)
        self.z = sigmoid(self.a)
        # fold in a bias in z
        self.z_bias =np.insert(self.z,0,1)

        self.b = linear(self.z_bias,self.w2)
        self.y_hat = softmax(self.b)


def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)

def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data *with the bias folded in*
    :param nn: neural network class
    :return: output probability
    """
    alpha = nn.w1
    nn.a = linear(X,alpha)
    nn.z = sigmoid(nn.a)
    nn.z_bias = np.insert(nn.z,0,1)
    beta = nn.w2
    nn.b = linear(nn.z_bias,beta)

    nn.y_hat = softmax(nn.b)
    return (nn.y_hat)

def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data *with the bias folded in*
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    gb = d_cross_entropy_vec(y, y_hat)
    # gz = gb*(db/dz)
    # gbeta = gb*(db/dbeta)
    dbdbeta = d_linear(nn.z_bias, nn.w2)[1]


    dbdbeta = dbdbeta.reshape(1,len(dbdbeta))

    gbcol = gb.reshape(len(gb),1)

    gbeta = (gbcol*dbdbeta)


    dbdz = d_linear(nn.z_bias, nn.w2)[0]

    gz = np.matmul(dbdz,gb)[1:]

    #ga = dz/da*dl/dz = dz/da*gz
    dzda = d_sigmoid(nn.z)



    ga = dzda*np.squeeze(gz)


    #galpha = dlda*dadalpha = ga*xt
    dadalpha = d_linear(X, nn.w1)[1]

    dadalpha = dadalpha.reshape(1,len(dadalpha))

    galpha = np.expand_dims(ga,axis=1)@dadalpha




    return (galpha,gbeta)


def test(xtest,ytest, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    length = xtest.shape[0]
    counter = 0
    for i in range (length):
        teste = xtest[i,:]
        yhat = forward(teste, nn)
        pre = yhat.argmax()
        if pre != ytest[i]:
            counter+=1
    return (counter/length)

def preres (x,nn,file):
    length = x.shape[0]
    for i in range (length):
        ex = x[i,:]
        yhat = forward(ex, nn)
        res = yhat.argmax()
        file.write(f'{res}\n')
        

def predict(x,y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    length = x.shape[0]
    ce = 0
    for i in range (length):
        teste = x[i,:]
        yhat = forward(teste, nn)
        realy = y[i]
        
        ce += cross_entropy(realy, yhat)
    return (ce/length)


def train(X_tr, y_tr, X_te, y_te, nn):

    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param X_te: train data
    :param y_te: train label
    :param nn: neural network class
    """
    alpha = nn.w1
    beta = nn.w2
    lr = nn.lr
    epochs = nn.n_epoch
    examples = X_tr.shape[0]
    for e in range(0,epochs):      
        Xtr = (shuffle(X_tr, y_tr, e))[0]
        ytr = (shuffle(X_tr, y_tr, e))[1]
        for i in range (examples):
            ex = Xtr[i,:]
            yl = ytr[i]
            o = forward(ex, nn)
            galpha,gbeta = backward(ex, yl, o, nn)
            
            grad_sum_w1 = np.square(galpha)
            grad_sum_w2 = np.square(gbeta)        
            nn.grad_sum_w1 += grad_sum_w1
            nn.grad_sum_w2 += grad_sum_w2
            lrw1 = (lr/np.sqrt(nn.grad_sum_w1+nn.epsilon1))
            lrw2 = (lr/np.sqrt(nn.grad_sum_w2+nn.epsilon2))
            neww1 = nn.w1 - lrw1*galpha
            neww2 = nn.w2 - lrw2*gbeta
            nn.w1 = neww1
            nn.w2 = neww2
        
        
        trainerror = predict(X_tr,y_tr, nn)
        metricfile.write(f'epoch={e+1} crossentropy(train): {trainerror}\n')
        validationerror = predict(X_te,y_te, nn)   
        metricfile.write(f'epoch={e+1} crossentropy(validation): {validationerror}\n')     
    return (nn)
            
input_size = len(xtrain[0,:])       
if (init_flag == 1):
    nn = NN(lr = lr,n_epoch= nepochs,weight_init_fn = random_init, input_size=input_size,hidden_size=hiddenunit,output_size = 10)
else:
    nn = NN(lr = lr,n_epoch= nepochs,weight_init_fn = zero_init, input_size=input_size,hidden_size=hiddenunit,output_size = 10)




train(xtrain, ytrain, xtest, ytest, nn)
errortrain = test(xtrain,ytrain, nn)
errortest = test(xtest,ytest, nn)

metricfile.write(f'error(train): {errortrain}\n')

metricfile.write(f'error(validation): {errortest}\n')

preres (xtrain,nn,trainfile)
preres (xtest,nn,testfile)