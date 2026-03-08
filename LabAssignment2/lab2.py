import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


######### Classes & Functions ##########

### Training functions

def train_step(model, x, y):
    # Given the model, training data (x), and labels (y), trains the model one step
    # Returns the loss value from this training step
    y_hat = model(x)
   
    # Loss and backprop - NOTE: Gradient is linear, so averaging loss = batch optimizer with size 1 is the same as one-by-one
    zeroGrad(model)
    loss = model.lossFn(y_hat, y).mean()
    loss.backward()
   
    # Optimizer update
    model.optimizer()

    # Find number of correctly labeled data for training dataset
    numTrainCorrect = (y_hat.argmax(dim=1) == y).sum().item()
           
    return loss.item(), numTrainCorrect

def trainAndTest(model, numEpochs, train_loader, test_loader, device = None):
    # Train and test model given various parameters.
    # Returns list of losses, training accuracies, test accuracies, and duration for each epoch.

    # Keep track of loss, accuracy, and comp time
    epochLosses = []
    epochTrainAccuracies = []
    epochTestAccuracies = []
    epochTimes = []
    print("Starting Training!")
    for epoch in range(numEpochs):
        model.train()

        # Keep track of loss, accuracy, and comp time
        epochLoss = 0.0
        epochTrainCorrect = 0
        epochTestCorrect = 0
        start = time.time()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{numEpochs} Training Data", leave=False):
            if device is not None:
                # Move data to GPU
                x = x.to(device)
                y = y.to(device)
            # Track total epoch loss and training accuracy over all images in batch
            loss, numTrainCorrect = train_step(model, x, y)
            epochLoss += loss*x.size(0)
            epochTrainCorrect += numTrainCorrect

        # Test model on test data
        model.eval()
        for x, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{numEpochs} Test Data", leave=False):
            if device is not None:
                # Move data to GPU
                x = x.to(device)
                y = y.to(device)
            with torch.no_grad():
                y_hat = model(x)
                epochTestCorrect += (y_hat.argmax(dim=1) == y).sum().item()
           

        # Record loss, accuracy, and time statistics
        epochTime = time.time() - start
        numTrainData = len(train_loader.dataset)
        avgLoss = epochLoss/numTrainData
        trainAcc = epochTrainCorrect/numTrainData
        testAcc = epochTestCorrect/len(test_loader.dataset)
        epochLosses.append(avgLoss)
        epochTrainAccuracies.append(trainAcc)
        epochTestAccuracies.append(testAcc)
        epochTimes.append(epochTime)
        print(f"Epoch {epoch+1}/{numEpochs} - Loss: {avgLoss:.4f} - Training Accuracy: {trainAcc*100:.4f}% - Testing Accuracy: {testAcc*100:.4f}% - Epoch Time: {epochTime:.4f} s")

    return epochLosses, epochTrainAccuracies, epochTestAccuracies, epochTimes

def trainingBenchmark(model, train_loader, device = None):
    # Train 1 epoch of a model and benchmark computation time distributions.
    # Prints time spent at each step in training

    # Keep track of loss, accuracy, and comp time
    print("Starting Training!")
    # Add timing variables
    transfer_time = 0
    forward_time = 0
    backward_time = 0
    optimizer_time = 0
    loader_time = 0

    model.train()
    epoch_start = time.time()
   
    iter_start = time.time()  # NEW: track time before first batch
    for x, y in tqdm(train_loader, desc=f"Benchmarking Epoch", leave=False):
        # Time loader (time since last iteration ended)
        loader_time += time.time() - iter_start  # NEW
       
        # Time transfers
        t0 = time.time()
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        transfer_time += time.time() - t0
       
        # Time forward pass
        t0 = time.time()
        y_hat = model(x)
        forward_time += time.time() - t0
       
        # Time backward pass
        t0 = time.time()
        zeroGrad(model)
        loss = model.lossFn(y_hat, y).mean()
        loss.backward()
        backward_time += time.time() - t0
       
        # Time optimizer
        t0 = time.time()
        model.optimizer()
        optimizer_time += time.time() - t0
       
        iter_start = time.time()  # Mark end of iteration
   
    # Print breakdown after first epoch
    total = transfer_time + forward_time + backward_time + optimizer_time + loader_time
    epoch_wall_time = time.time() - epoch_start
    print(f"\nTime breakdown (epoch 1):")
    print(f"  Loader:    {loader_time:6.2f}s ({100*loader_time/total:5.1f}%)")
    print(f"  Transfer:  {transfer_time:6.2f}s ({100*transfer_time/total:5.1f}%)")
    print(f"  Forward:   {forward_time:6.2f}s ({100*forward_time/total:5.1f}%)")
    print(f"  Backward:  {backward_time:6.2f}s ({100*backward_time/total:5.1f}%)")
    print(f"  Optimizer: {optimizer_time:6.2f}s ({100*optimizer_time/total:5.1f}%)")
    print(f"  Total:     {total:6.2f}s")
    print(f"  Wall time: {epoch_wall_time:6.2f}s (includes tqdm overhead)\n")

### Activation Functions

def softmax(x: torch.Tensor, dim=1) -> torch.Tensor:
    # Softmax activation function
    # For numerical stability, multiply both the numerator and denominator by e^-x_max, where x_max is the max value in each batch
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    expx = torch.exp(x-x_max)
    return expx/torch.sum(expx, dim=1, keepdim=True)

def crossEntropy(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Returns the cross entropy loss of x.
    # Clamp to prevent log(0) or log(1)
    y_hat = torch.clamp(y_hat, min=1e-8, max=1.0 - 1e-8)
    # Make y_true a column vector
    y_true = y_true.view(-1,1)
    return -torch.log(y_hat.gather(dim=1, index=y_true))

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    # Sigmoid activation function
    return 1.0/(1.0 + torch.exp(-x))

def leakyReLU(x: torch.Tensor) -> torch.Tensor:
    # Leaky ReLU activation function
    return torch.max(x, 0.1*x)

def SiLU(x: torch.Tensor) -> torch.Tensor:
    # SiLU activation function
    return x/(1 + torch.exp(-x))

### Zero Grad

def zeroGrad(model):
    # Given model, zeros all parameters
    for param in model.parameters():
        param.grad = None

### Optimizer Classes

class Optimizer():
    # Base optimizer class.
    # NOTE: It is assumed the gradients were calculated from averaged batch losses. For one-by-one sample processing, batch size = 1
    def __init__(self, model, learningRate):
        self.model = model
        self.learningRate = learningRate

class SGD(Optimizer):
    def __call__(self):
        # SGD Batch Optimizer given PyTorch model and learning rate
        with torch.no_grad(): # Disable tracking during parameter update
            for param in self.model.parameters():
                param -= self.learningRate*param.grad

class SGDMomentum(Optimizer):
    def __init__(self, model, learningRate, alpha):
        # alpha is the constant that controls momentum strength
        super().__init__(model, learningRate)
        self.alpha = alpha
        # Variables to store momentum from previous step
        self.m = {}

    def __call__(self):
        # SGD Optimizer WITH MOMENTUM given PyTorch model and learning rate
        with torch.no_grad():
            for param in self.model.parameters():
                if param not in self.m:
                    # First iteration
                    self.m[param] = param.grad
                else:
                    # Normal iteration
                    self.m[param] = self.alpha*self.m[param] + param.grad

                # Update
                param -= self.learningRate*self.m[param]
       


### NN Classes

class SimpleTwoLayerNN(nn.Module):
    def __init__(self, lossFn = crossEntropy, optCls = SGD, learningRate = 0.1, inputSize = 3072, hiddenSize = 512, numClasses = 100):
        # Initial input size is 32*32*3 = 3072. Hidden size was chosen somewhat arbritrarily. There are 100 classes in the CIFAR-100
        super(SimpleTwoLayerNN, self).__init__()
        self.lossFn = lossFn
        self.optimizer = optCls(self, learningRate)
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses

        # Initialize Layers
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, numClasses)

    def forward(self, x):
        # Flatten input data so that it becomes a 2D tensor -> [Batch, Image Data]
        x = x.flatten(start_dim = 1)

        # Feedforward Architecture
        p1 = self.linear1(x)
        o1 = sigmoid(p1)
        p2 = self.linear2(o1)
        output = softmax(p2)

        return output
   
class BaselineDeepNN(nn.Module):
    def __init__(self, actFn = sigmoid, lossFn = crossEntropy, optCls = SGD, learningRate = 0.1, optKwargs = {}, numClasses = 100, nInChannels = 3, nChannel1 = 32, nChannel2 = 64, nChannel3 = 128, hSize = 512):
        # Inputs:
        #   actFn = Desired activation function for all layers except output, which is softmax
        #   lossFn = Desired loss function
        #   optCls = Desired optimizer class
        #   learningRate = Learning rate of optimizer
        #   optKwargs = Additional optimizer keyword arguments
        #   numClasses = Number of classes for output
        #   nInChannels = Number of channels in
        #   nChannel = Number of channels in hidden layers
        #   hSize = Number of hidden neurons between fully connected layers
        super(BaselineDeepNN, self).__init__()
        self.actFn = actFn
        self.lossFn = lossFn
        self.optimizer = optCls(self, learningRate, **optKwargs)

        # MaxPool function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 1: Convolution - 3 input channels for RGB -> MaxPool (32x32 -> 16x16)
        self.conv1 = nn.Conv2d(in_channels=nInChannels, out_channels=nChannel1, kernel_size=3, stride=1, padding=1)

        # Layer 2: Convolution -> MaxPool (16x16 -> 8x8)
        self.conv2 = nn.Conv2d(in_channels=nChannel1, out_channels=nChannel2, kernel_size=3, stride=1, padding=1)

        # Layer 3: Convolution
        self.conv3 = nn.Conv2d(in_channels=nChannel2, out_channels=nChannel3, kernel_size=3, stride=1, padding=1)

        # Layer 4: Linear Layer - Feature map is 8x8 at this point
        self.lin1 = nn.Linear(in_features=nChannel3*8*8,out_features=hSize)

        # Layer 5: Linear Layer
        self.lin2 = nn.Linear(in_features=hSize, out_features=numClasses)

    def forward(self, x):
        # Let values z be after layer, p be after activation, and o be after pooling or other processing, if needed

        # Layer 1: Convolution - 3 input channels for RGB -> MaxPool (32x32 -> 16x16)
        z1 = self.conv1(x)
        p1 = self.actFn(z1)
        o1 = self.pool(p1)

        # Layer 2: Convolution -> MaxPool (16x16 -> 8x8)
        z2 = self.conv2(o1)
        p2 = self.actFn(z2)
        o2 = self.pool(p2)

        # Layer 3: Convolution
        z3 = self.conv3(o2)
        p3 = self.actFn(z3)
        # Flatten data
        o3 = p3.flatten(start_dim=1)

        # Layer 4: Linear Layer
        z4 = self.lin1(o3)
        p4 = self.actFn(z4)

        # Layer 5: Linear Layer
        z5 = self.lin2(p4)
        output = softmax(z5)

        return output

########## Main Function ##########

def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    ### Data loading and preprocessing
    # Basic preprocessing
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Basically image translation/cutoff
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( # Use standard values for CIFAR100
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])
    # No data augmentation for test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( # Use standard values for CIFAR100
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    # Download and load datasets
    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    # One-by-one DataLoader
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    ### Initialize and train two layer model to verify dataset
    # numEpochs = 10
    # learningRate = 0.001
    # TwoLNN = SimpleTwoLayerNN(lossFn = crossEntropy, optCls = SGD, learningRate = learningRate, inputSize = 3072, hiddenSize = 512, numClasses = 100)
    # trainAndTest(TwoLNN, numEpochs, train_loader, test_loader)

    ### Initialize and train baseline model on dataset
    numEpochs = 10
    learningRate = 0.05
    # BaselineNN = BaselineDeepNN(sigmoid, crossEntropy, SGD, learningRate, numClasses = 100, nInChannels = 3, nChannel1 = 32, nChannel2 = 64, nChannel3 = 128, hSize = 512)
    # trainAndTest(BaselineNN, numEpochs, train_loader, test_loader)

    ### Activation Functions
    ## Leaky ReLU
    LeakyReLUNN = BaselineDeepNN(leakyReLU, crossEntropy, SGD, learningRate, numClasses = 100, nInChannels = 3, nChannel1 = 32, nChannel2 = 64, nChannel3 = 128, hSize = 512).to(device)
    # trainAndTest(LeakyReLUNN, numEpochs, train_loader, test_loader, device)
    # trainingBenchmark(LeakyReLUNN, train_loader, device)

    ## Leaky SiLU
    # SiLUNN = BaselineDeepNN(SiLU, crossEntropy, SGD, learningRate, numClasses = 100, nInChannels = 3, nChannel1 = 32, nChannel2 = 64, nChannel3 = 128, hSize = 512)
    # trainAndTest(SiLUNN, numEpochs, train_loader, test_loader)

    ### Optimizers
    ## Mini-batch - Just SGD but with different batch size DataLoader
    # Batch DataLoader
    batch_size = 128
    train_loader_batch = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader_batch  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # trainAndTest(LeakyReLUNN, numEpochs, train_loader_batch, test_loader_batch, device)

    ## Mini-Batch SGD with Momentum
    LeakyReLUMomentumSGD = BaselineDeepNN(leakyReLU, crossEntropy, SGDMomentum, learningRate, optKwargs = {'alpha': 0.9}, numClasses = 100, nInChannels = 3, nChannel1 = 32, nChannel2 = 64, nChannel3 = 128, hSize = 512).to(device)
    trainAndTest(LeakyReLUMomentumSGD, numEpochs, train_loader_batch, test_loader_batch, device)

if __name__ == "__main__":
    main()