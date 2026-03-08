import torch
import math
import numpy as np
import matplotlib.pyplot as plt

## User-defined functions and classes

def plotBinary(x, y, title = ""):
    # Plots a binary set of data, where x are 2d data points and y are the corresponding labels (0 or 1).
    # Does NOT show the plots
    plt.figure(figsize=(6, 6))
    plt.scatter(x[(y == 0).squeeze(),0], x[(y == 0).squeeze(),1], color='red', label="Label = 0")
    plt.scatter(x[(y == 1).squeeze(),0], x[(y == 1).squeeze(),1], color='blue', label="Label = 1")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.legend()
    plt.xlim(x[:,0].min().item() - 0.2, x[:,0].max().item() + 0.2)
    plt.ylim(x[:,1].min().item() - 0.2, x[:,1].max().item() + 0.2)

def benchmarkModel(x_data, y_data, hiddenDim=2, beta=0.1, numAttempts=10, numEpochs=1000, returnConvergence = False, weightFunc = None):
    """
    Benchmarks a two-layer network on a given dataset. weightFunc is a function determining the initialization of weights and biases. weightFunc returns [w1,b1,w2,b2]

    Returns:
        perfectCount = Number of runs achieving 100% accuracy
        finalAcc = Array of final accuracy of each attempt
        bestModel = Best-performing trained model
        returnConvergence = Flag to see if func should return convergence times
    """

    # Zero bias initialization with d^(-1/2) default variance
    if weightFunc is None:
        weightFunc = lambda hiddenDim, inputDim: (
            torch.randn(hiddenDim, inputDim) * (inputDim ** -0.5), # w1
            torch.zeros(hiddenDim), # b1
            torch.randn(hiddenDim) * (hiddenDim ** -0.5), # w2
            0.0 # b2
        )
    
    inputDim = x_data.size(1)
    perfectCount = 0
    finalAcc = []
    bestEpoch = numEpochs + 1
    bestAcc = 0.0
    bestModel = None
    convergenceTimes = []
    for i in range(numAttempts):
        w1, b1, w2, b2 = weightFunc(hiddenDim, inputDim)
        model = TwoLayerNetwork(w1, b1, w2, b2, beta)

        # Train model
        for j in range(numEpochs):
            for x, y in zip(x_data, y_data):
                model.infer(x)
                model.update(y.item())
            for x, y in zip(x_data, y_data):
                model.infer(x)
                model.updateAccuracies(y.item())
            model.nextEpoch()
        finalAcc.append(model.accuracies[-1])
        convergenceTimes.append(model.convergence)

        # Perfect accuracy case
        if finalAcc[-1] == 1.0:
            perfectCount += 1
            firstPerfectEpoch = next(i for i, a in enumerate(model.accuracies) if a == 1.0)
            if firstPerfectEpoch < bestEpoch:
                bestEpoch = firstPerfectEpoch
                bestAcc = 1.0
                bestModel = model
        # Best non-perfect model
        elif bestAcc < 1.0 and finalAcc[-1] > bestAcc:
            bestAcc = finalAcc[-1]
            bestModel = model
    if returnConvergence:
        return perfectCount, finalAcc, bestModel, convergenceTimes
    else:
        return perfectCount, finalAcc, bestModel

def injectNoise(x, r):
    # Injects noise from a normal distribution with standard deviation r into the data x
    return x + torch.randn_like(x)*r

class Perceptron:
    # Class for perceptron
    def __init__(self, weights: torch.Tensor, bias: float, beta: float):
        # Takes initial weights, bias, and learning rate (beta) terms as arguments
        self.w = weights # Weights vector
        self.b = bias # Bias term
        self.beta = beta # Learning rate
        self.x = torch.zeros(weights.size(0)) # Inputs vector
        self.y = 0 # Output corresponding to x
        self.numCorrect = 0 # Number of correct labels this epoch
        self.numTotal = 0
        self.accuracies = []
        self.decisionAngles = [] # IF input dimensions are 2d

    def infer(self, x: torch.Tensor):
        # Inference rule of perceptron
        self.x = x
        z = torch.dot(self.w, self.x) + self.b
        # Sign activation function
        if z >= 0:
            self.y = 1
        else:
            self.y = 0
        return self.y
        
    def update(self, y_truth):
        # Update weights
        dy = self.y - y_truth # Difference in predicted vs expected
        self.b = self.b - self.beta*dy
        self.w = self.w - self.beta*dy*self.x
    
    def updateAccuracies(self, y_truth):
        # Updates accuracy parameters
        self.numTotal += 1
        if self.y == y_truth:
            self.numCorrect += 1
    
    def nextEpoch(self):
        # Record accuracy after each epoch. Also, reset numCorrect and numTotal
        self.accuracies.append(self.numCorrect/self.numTotal)
        if self.w.size(0) == 2:
            self.decisionAngles.append(math.atan2(self.w[1].item(), self.w[0].item()) + math.pi/2)
        self.numCorrect = 0
        self.numTotal = 0

    def boundaryLineFunc(self, x1):
        # For 2 input perceptron, calculates x2 points corresponding to x1 based on current weights:
        # w1*x1 + w2*x2 + b = 0
        return -(self.w[0].item()*x1 + self.b)/self.w[1].item()
    
    def plotTrainingStats(self, title):
        numEpochs = np.arange(1, len(self.accuracies) + 1)

        plt.figure(figsize=(6, 6))

        # Accuracy (left y-axis)
        ax1 = plt.gca()
        ax1.plot(numEpochs, self.accuracies, label="Model Accuracy", color="blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Model Accuracy")
        ax1.set_ylim(0, 1)
        ax1.grid(True)

        # Decision boundary angle (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(numEpochs, self.decisionAngles, label="Decision Boundary Angle", color="red", linestyle="-.")
        ax2.set_ylabel("Decision Boundary Angle (rad)")

        # Legend and title
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title(title)

class TwoLayerNetwork:
    # Hidden layer class
    def __init__(self, w1, b1, w2, b2, beta: float):
        # Takes initial weights, bias, and learning rate (beta) terms as arguments
        self.w1 = w1 # Weights for hidden layer
        self.b1 = b1 # Bias for hidden layer
        self.w2 = w2 # Weights for output layer
        self.b2 = b2 # Bias for output layer
        self.beta = beta # Learning rate
        self.x = torch.zeros(w1.size(1)) # Inputs vector
        self.y = 0.0
        self.numCorrect = 0 # Number of correct labels this epoch
        self.numTotal = 0
        self.accuracies = []
        self.convergence = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + torch.exp(-z))
    
    def dsigmoid(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    def infer(self, x: torch.Tensor):
        # Inference rule of two layer network
        self.x = x
        self.p1 = torch.matmul(self.w1, self.x) + self.b1
        self.o1 = self.sigmoid(self.p1)
        self.p2 = torch.dot(self.w2, self.o1) + self.b2
        self.y = self.sigmoid(self.p2)

        return self.y
        
    def update(self, y_truth):
        # Change in weights and biases (backpropagation)
        dy = self.y - y_truth
        dp2 = dy*self.dsigmoid(self.p2)
        db2 = dp2
        dw2 = dp2*self.o1
        do1 = dp2*self.w2
        dp1 = do1*self.dsigmoid(self.p1)
        db1 = dp1
        dw1 = torch.outer(dp1, self.x)
        # Update weights and biases
        self.w2 -= self.beta * dw2
        self.b2 -= self.beta * db2
        self.w1 -= self.beta * dw1
        self.b1 -= self.beta * db1

    def updateAccuracies(self, y_truth):
        # Update accuracy parameters
        self.numTotal += 1
        # Since y is not a step function, follow best binary guess for y
        y_pred = 1 if self.y >= 0.5 else 0
        if y_pred == y_truth:
            self.numCorrect += 1

    def nextEpoch(self):
        # Record accuracy after each epoch. Also, reset numCorrect and numTotal
        self.accuracies.append(self.numCorrect/self.numTotal)
        # Record convergence epoch, if applicable
        if self.accuracies[-1] == 1.0 and self.convergence is None:
            self.convergence = len(self.accuracies)
        self.numCorrect = 0
        self.numTotal = 0

    def plotTrainingStats(self, title, newFigure=True, plotLabel="Model Accuracy", plotColor = "blue"):
        numEpochs = np.arange(1, len(self.accuracies) + 1)

        if newFigure:
            plt.figure(figsize=(6, 6))

        # Accuracy
        ax1 = plt.gca()
        ax1.plot(numEpochs, self.accuracies, label=plotLabel, color=plotColor)
        if newFigure:
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Model Accuracy")
            ax1.set_ylim(0, 1)
            ax1.grid(True)
            ax1.legend(loc="best")

            plt.title(title)
    
    def plotBoundary(self, numTestPts = 400, title = ""):
        # Generate test points
        S = torch.linspace(-1.0, 1.0, int(math.sqrt(numTestPts)))
        testPts = torch.cartesian_prod(S, S)
        y_preds = []
        for pt in testPts:
            # Since y is not a step function, follow best binary guess for y
            y_pred = 1 if self.infer(pt) >= 0.5 else 0
            y_preds.append(y_pred)
        plotBinary(testPts, torch.tensor(y_preds), title)


### ====================================== Main body ========================================== ##

torch.manual_seed(1)

### 1) Generating Datasets & Perceptron

# Filepath to hw1 folder
fp = "/Users/Jacob/Downloads/School/CSCI 5922/HW/HW1/"

## 1a) Linear Dataset

# Generate test data points
S_lin = torch.arange(-1.0,1.5,0.5)
x_lin = torch.cartesian_prod(S_lin, S_lin) # Cartesian product gives all test points

# Generate labels associated with test points
y_lin = (x_lin[:,1] > x_lin[:,0]).int().view(-1,1)

# Plotting
plotBinary(x_lin, y_lin, "Plot of Linear Boundary Dataset")
plt.savefig(fp + 'linearData.png')
plt.close()

## 1b) XOR Dataset

# Generate test data points
S_XOR = torch.tensor([1.0,-1.0])
x_XOR = torch.cartesian_prod(S_XOR,S_XOR)

# Generate labels associated with test points
y_XOR = (x_XOR[:,1] != x_XOR[:,0]).int().view(-1,1)

# Plotting
plotBinary(x_XOR, y_XOR, "Plot of XOR Dataset")
plt.savefig(fp + 'XORData.png')
plt.close()

### Perceptron Training

numEpochs = 100

# Initialize perceptrons
inputDim = x_lin.size(1) # Note: There are 2 inputs for both linear boundary and XOR datasets
sigma = 1/math.sqrt(inputDim)

pTronLin = Perceptron(torch.randn(inputDim)*sigma, 0.0, 0.001)
pTronXOR = Perceptron(torch.randn(inputDim)*sigma, 0.0, 0.1)

# Train perceptrons
for i in range(numEpochs):
    # Training
    for x,y in zip(x_lin,y_lin):
        pTronLin.infer(x)
        pTronLin.update(y.item())
    for x,y in zip(x_XOR,y_XOR):
        pTronXOR.infer(x)
        pTronXOR.update(y.item())
    # Benchmarking
    for x,y in zip(x_lin,y_lin):
        pTronLin.infer(x)
        pTronLin.updateAccuracies(y.item())
    for x,y in zip(x_XOR,y_XOR):
        pTronXOR.infer(x)
        pTronXOR.updateAccuracies(y.item())
    pTronLin.nextEpoch()
    pTronXOR.nextEpoch()

# Plot model accuracy & decision boundary angle vs epoch
pTronLin.plotTrainingStats("Perceptron Training Performance for Linear Boundary Data")
plt.savefig(fp + 'epochLin.png')
plt.close()
pTronXOR.plotTrainingStats("Perceptron Training Performance for XOR Data")
plt.savefig(fp + 'epochXOR.png')
plt.close()

# Plot learned decision boundary on data
x1 = np.linspace(-1.5, 1.5, 100)
# Linear data
x2 = pTronLin.boundaryLineFunc(x1)
plotBinary(x_lin, y_lin, "Plot of Learned Boundary with Linear Boundary Dataset")
plt.plot(x1, x2)
plt.savefig(fp + 'linearBoundary.png')
plt.close()
# XOR data
x2 = pTronXOR.boundaryLineFunc(x1)
plotBinary(x_XOR, y_XOR, "Plot of Learned Boundary with XOR Dataset")
plt.plot(x1, x2)
plt.savefig(fp + 'XORBoundary.png')
plt.close()

### 2) Multi-Layer Models

# Initialize and benchmark NNs
beta = 0.1
numAttempts = 10
numEpochs = 1000
hiddenDim = 2

perfectLin, _, bestLinModel = benchmarkModel(x_lin, y_lin, hiddenDim, beta, numAttempts, numEpochs)
perfectXOR, _, bestXORModel = benchmarkModel(x_XOR, y_XOR, hiddenDim, beta, numAttempts, numEpochs)

# Report results & plot model accuracy vs epoch
print("Number of perfect accuracy models on Linear Dataset: ", perfectLin)
print("Number of perfect accuracy models on XOR Dataset: ", perfectXOR)
bestLinModel.plotTrainingStats("Two Layer NN Training Performance for Linear Boundary Data")
plt.savefig(fp + '2LayerEpochLin.png')
plt.close()
bestXORModel.plotTrainingStats("Two Layer NN Training Performance for XOR Data")
plt.savefig(fp + '2LayerEpochXOR.png')
plt.close()
bestLinModel.plotBoundary(400, "Two Layer NN for Linear Boundary Decision Boundary")
plt.savefig(fp + '2LayerLinDecisionBound.png')
plt.close()
bestXORModel.plotBoundary(400, "Two Layer NN for XOR Decision Boundary")
plt.savefig(fp + '2LayerXORDecisionBound.png')
plt.close()

### 3) Noisy XOR
# Expanded datasets
x_XOR_Expanded = x_XOR.repeat(10,1)
y_XOR_Expanded = y_XOR.repeat(10,1)

# Noisy x_XOR datasets
x_XOR_25 = injectNoise(x_XOR_Expanded, 0.25)
x_XOR_50 = injectNoise(x_XOR_Expanded, 0.50)
x_XOR_75 = injectNoise(x_XOR_Expanded, 0.75)

# Plot noisy XOR datasets
plotBinary(x_XOR_25, y_XOR_Expanded, "Plot of Noisy XOR (r = 0.25) Dataset")
plt.savefig(fp + 'XOR25Data.png')
plt.close()
plotBinary(x_XOR_50, y_XOR_Expanded, "Plot of Noisy XOR (r = 0.50) Dataset")
plt.savefig(fp + 'XOR50Data.png')
plt.close()
plotBinary(x_XOR_75, y_XOR_Expanded, "Plot of Noisy XOR (r = 0.75) Dataset")
plt.savefig(fp + 'XOR75Data.png')
plt.close()

# Benchmark the two layer model for number of perfect accuracy attempts
numInstances = 100
_, accXOR25, bestXOR25Model = benchmarkModel(x_XOR_25, y_XOR_Expanded, hiddenDim, beta, numInstances, numEpochs)
_, accXOR50, bestXOR50Model = benchmarkModel(x_XOR_50, y_XOR_Expanded, hiddenDim, beta, numInstances, numEpochs)
_, accXOR75, bestXOR75Model = benchmarkModel(x_XOR_75, y_XOR_Expanded, hiddenDim, beta, numInstances, numEpochs)

# Plot accuracy data
plt.figure(figsize=(6, 6))
bp = plt.boxplot([accXOR25, accXOR50, accXOR75], patch_artist=True)
# Define colors for each box
colors = ['blue', 'red', 'green']
# Apply colors to the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
# Add title and labels
plt.title("Accuracy Distribution of Two Layer Model on Noisy XOR Data")
plt.ylabel("Final Accuracy")
plt.xlabel("Input Noise Standard Deviation (r)")
# Set x-axis tick labels
plt.xticks([1, 2, 3], ['0.25', '0.50', '0.75'])
plt.savefig(fp + "NoisyXORBoxplot.png")
plt.close()

# Plot best model decision boundaries

bestXOR25Model.plotBoundary(400, "Plot of Noisy XOR (r = 0.25) Decision Boundary")
plt.savefig(fp + 'XOR25Boundary.png')
plt.close()
bestXOR50Model.plotBoundary(400, "Plot of Noisy XOR (r = 0.5) Decision Boundary")
plt.savefig(fp + 'XOR50Boundary.png')
plt.close()
bestXOR75Model.plotBoundary(400, "Plot of Noisy XOR (r = 0.75) Decision Boundary")
plt.savefig(fp + 'XOR75Boundary.png')
plt.close()

### 4) High Dimensional Parity
# Create higher dimensional parity labeled datasets
S = torch.tensor([-1.0, 1.0])
xDim = {}
yDim = {}
modelsDimTwo = {} # Models with hidden layer dimension 2
modelsDimIn = {} # Models with hidden layer dimension d_input
for dim in [4,6,8]:
    xDim[dim] = torch.cartesian_prod(*([S] * dim))
    # Check for even number of -1
    yDim[dim] = ((xDim[dim] == -1.0).sum(dim=1)%2).view(-1,1)
    _, _, modelsDimTwo[dim] = benchmarkModel(xDim[dim],yDim[dim],2,beta,1,numEpochs)
    _, _, modelsDimIn[dim] = benchmarkModel(xDim[dim],yDim[dim],dim,beta,1,numEpochs)
    modelsDimTwo[dim].plotTrainingStats("Two Layer NN Training Performance for Parity Data d_input = " + str(dim), plotLabel="d_hidden = 2")
    modelsDimIn[dim].plotTrainingStats("", newFigure = False, plotLabel="d_hidden = " + str(dim), plotColor="red")
    plt.legend()
    plt.savefig(fp + 'XORDim' + str(dim))
    plt.close()

modelsDimTwo[4].plotTrainingStats("Two Layer NN Training Performance for Parity Data d_hidden = 2", plotLabel="d_input = 4")
modelsDimTwo[6].plotTrainingStats("", newFigure = False, plotLabel="d_input = 6", plotColor="red")
modelsDimTwo[8].plotTrainingStats("", newFigure = False, plotLabel="d_input = 8", plotColor="green")
plt.legend()
plt.savefig(fp + 'XORHiddenDim2.png')
plt.close()

modelsDimIn[4].plotTrainingStats("Two Layer NN Training Performance for Parity Data d_hidden = d_input", plotLabel="d_input = 4")
modelsDimIn[6].plotTrainingStats("", newFigure = False, plotLabel="d_input = 6", plotColor="red")
modelsDimIn[8].plotTrainingStats("", newFigure = False, plotLabel="d_input = 8", plotColor="green")
plt.legend()
plt.savefig(fp + 'XORHiddenDimEqual.png')
plt.close()

### Extra Credit

# Given Params
beta = 0.1
numEpochs = 500
numInstances = 100

# Standard Two-Layer Model Training
_, _, _, convergenceTimesNorm = benchmarkModel(x_lin, y_lin, hiddenDim, beta, numInstances, numEpochs, True)

# Uniform Two-Layer Model Training
weightFuncUnif = lambda hiddenDim, inputDim: (
    (2.0 * torch.rand(hiddenDim, inputDim) - 1.0) / (10.0 * inputDim**0.5), # w1 ~ U[-1/(10√d_in), 1/(10√d_in)]
    torch.zeros(hiddenDim), # b1
    (2.0 * torch.rand(hiddenDim) - 1.0) / (10.0 * hiddenDim**0.5), # w2 ~ U[-1/(10√d_h), 1/(10√d_h)]
    0.0 # b2
)
_, _, _, convergenceTimesUnif = benchmarkModel(x_lin, y_lin, hiddenDim, beta, numInstances, numEpochs, True, weightFuncUnif)

# Ones Two-Layer Model Training
weightFuncOnes = lambda hiddenDim, inputDim: (
    torch.ones(hiddenDim, inputDim), # w1 ~ U[-1/(10√d_in), 1/(10√d_in)]
    torch.zeros(hiddenDim), # b1
    torch.ones(hiddenDim), # w2 ~ U[-1/(10√d_h), 1/(10√d_h)]
    0.0 # b2
)
_, _, _, convergenceTimeOnes = benchmarkModel(x_lin, y_lin, hiddenDim, beta, 1, numEpochs, True, weightFuncOnes)

# Plot Convergence Times
plt.figure(figsize=(6, 6))
bp = plt.boxplot([convergenceTimesNorm, convergenceTimesUnif, convergenceTimeOnes], patch_artist=True)
# Define colors for each box
colors = ['blue', 'red', 'green']
# Apply colors to the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
# Add title and labels
plt.title("Convergence Time Distribution of Two Layer Model")
plt.ylabel("Convergence Time")
plt.xlabel("Initialization Weight Type")
# Set x-axis tick labels
plt.xticks([1, 2, 3], ['Gaussian', 'Uniform', 'Constant (Ones)'])
plt.savefig(fp + "ConvergenceBoxplot.png")
plt.close()

# Handpicked Two-Layer Model
weightFuncPicked = lambda hiddenDim, inputDim: (
    torch.tensor([[1.0, 1.0],[1.0,1.0]]), # w1
    torch.zeros(hiddenDim), # b1
    torch.tensor([0.16, 0.16]), # w2
    0.0 # b2
)
_, _, convergenceModel, convergenceTimePicked = benchmarkModel(x_lin, y_lin, hiddenDim, beta, 1, numEpochs, True, weightFuncPicked)

print(convergenceTimePicked[0])