# EE-399-HW4
**Author: Qiyue Chen**

**Date: 5/7/2023**

**Course: SP 2023 EE399**

## Abstract
This homework assignment consists of two parts. In the first part, we are provided with a series of data:

```python
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

Our task is to train a simple neural network to predict a series of numbers. We will split the data into training and testing sets and evaluate the performance of the model.

In the second part, we will perform Principal Component Analysis (PCA) on a dataset of handwritten digits from the MNIST database. We will train a neural network to classify the digits using reduced dimensionality data. Additionally, we will experiment with different neural network architectures and hyperparameters to improve the performance.

## Introduction and Overview
This assignment focuses on working with two datasets and applying neural networks to them.

The first dataset contains 31 data points. We will fit the data to a three-layer feed-forward neural network. The dataset will be split into training and testing sets, and the model's performance will be evaluated using the least square error.

The second dataset is the MNIST dataset, which consists of handwritten digits. We will compute the first 20 Principal Component Analysis (PCA) modes of the images and build a feed-forward neural network to classify the digits. Additionally, we will compare the results of the neural network with other classifiers such as LSTM, SVM, and decision trees.

## Theoretical Background
The key concept involved in this assignment is the feedforward neural network. A feedforward neural network is a mechanism where input signals are fed forward into the network, passing through different layers. The network produces outputs in the form of classifications at the output layer. Here is an animation illustrating a feedforward neural network:

![nagesh-pca-1](https://vitalflux.com/wp-content/uploads/2020/10/feed_forward_neural_network-1.gif)

1. **Layers**: The animation shows a neural network with four layers: one input layer, two hidden layers, and one output layer.
2. **Input fed into input layer**: There are four input variables fed into different nodes in the neural network through the input layer.
3. **Activations in the hidden layers**: The sum of input signals combined with weights and bias elements are fed into the neurons of the hidden layers. Each node adds all incoming values together and processes them with an activation function.
4. **Output in the final layer**: The activation signals from the hidden layers are combined with weights and fed into the output layer. At each node, all incoming values are added together and processed with a function to output probabilities.

There are several other important concepts that were introduced in previous assignments. Let's provide a brief explanation of these concepts:

1. Principle Component Analysis (PCA): PCA is a dimensionality reduction technique that transforms a dataset of potentially correlated variables into a set of linearly uncorrelated variables called principal components. It helps to capture the most important patterns or features in the data. PCA can be useful for reducing the dimensionality of high-dimensional data while retaining as much information as possible.

2. Support Vector Machines (SVM): SVM is a supervised learning algorithm that can be used for both classification and regression tasks. It finds a hyperplane in a high-dimensional space that maximally separates the classes of data points. SVM aims to find the best decision boundary that separates the data into different classes with the largest possible margin. It is particularly effective in cases where the data is not linearly separable by mapping the data into a higher-dimensional feature space.

3. Decision Trees: A decision tree is a flowchart-like structure where each internal node represents a feature or attribute, each branch represents a decision rule, and each leaf node represents the outcome. It is a simple yet powerful supervised learning algorithm used for classification and regression tasks. Decision trees are built by recursively partitioning the data based on the values of the input features to maximize the information gain at each node. They provide an interpretable and easily understandable model for decision-making.

These concepts have been covered in previous assignments, and they have their own advantages and applications in machine learning. It's important to understand these concepts and when to apply them based on the problem at hand.

## Algorithm Implementation and Development 

### Problem 1
#### Part(i) Fit the data to a three layer feed forward neural network.

```python
# Problem I
# Part (i)
# Define the data
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

Xmean, Xstd  = X.mean(), X.std()
Ymean, Ystd  = Y.mean(), Y.std()

X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd

# Split the data into training and test sets
X_train, Y_train = X[:20], Y[:20]
X_test, Y_test = X[20:], Y[20:]
```
The given code snippet performs data preprocessing and splitting for a machine learning problem. It starts by defining the independent variable `X` and the dependent variable `Y`. Then, it standardizes the data by subtracting the mean and dividing by the standard deviation for both `X` and `Y`. Finally, it splits the standardized data into training and test sets using array slicing. The first 20 elements are assigned to the training set, while the remaining elements are assigned to the test set. This preprocessing and splitting prepare the data for further machine learning tasks.

```python
# Convert data to PyTorch tensors
X_train_tensor = Variable(torch.Tensor(X_train).unsqueeze(1))
Y_train_tensor = Variable(torch.Tensor(Y_train).unsqueeze(1))
X_test_tensor = Variable(torch.Tensor(X_test).unsqueeze(1))
Y_test_tensor = Variable(torch.Tensor(Y_test).unsqueeze(1))

# Define the three layer neural network model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
          nn.Linear(1, 1024),
          nn.ReLU(),
          nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# Create an instance of the model
net = Net().to(device)

# Define the loss function
criterion = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
```
The given code snippet performs the following tasks:

1. Converts the training and test data (`X_train`, `Y_train`, `X_test`, `Y_test`) to PyTorch tensors using the `torch.Tensor()` function and reshapes them to have an additional dimension using `unsqueeze(1)`.

2. Defines a three-layer neural network model using the `torch.nn.Module` class. The model consists of two linear layers with ReLU activation in between. The input size of the first linear layer is 1, and the output size is 1024. The output size of the second linear layer is 1.

3. Creates an instance of the defined neural network model (`Net`) and moves it to the specified device (e.g., GPU) using the `to()` method.

4. Defines the loss function for the model as the Mean Squared Error (MSE) loss using `torch.nn.MSELoss()`.

5. Defines the optimizer to update the parameters of the neural network model using the Adam optimizer with a learning rate of 0.01. The optimizer is initialized with the parameters of the `net` model (`net.parameters()`).

In summary, the code prepares the data by converting it into PyTorch tensors, defines a neural network model with three layers, initializes the model instance, specifies the loss function (MSE), and sets up the optimizer (Adam) for training the model.

#### Part ii) Using the first 20 data points as training data, fit the neural network. Compute the least-square error for each of these over the training points. Then compute the least square error of these models on the test data which are the remaining 10 data points.

```python
# Part (ii)
# Train the model
num_epochs = 10000
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = net(X_train_tensor.unsqueeze(1).to(device))
    loss = criterion(outputs, Y_train_tensor.unsqueeze(1).to(device))

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute training and test loss for monitoring
    train_losses.append(loss.item())
    test_outputs = net(X_test_tensor.unsqueeze(1).to(device))
    test_loss = criterion(test_outputs, Y_test_tensor.unsqueeze(1).to(device))
    test_losses.append(test_loss.item())

    # Print progress
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}
```
This code trains a neural network model using a specified number of epochs. It performs the following steps:

1. Initializes empty lists to store training and test losses.

2. Enters a loop for each epoch and performs the following operations:

   a. Computes the forward pass by passing the training data through the neural network model.
   
   b. Calculates the loss between the predicted outputs and the training labels.
   
   c. Backpropagates the loss and updates the model's parameters using the optimizer.
   
   d. Computes the loss for both the training and test sets and appends them to their respective lists.
   
   e. Prints the training and test loss every 1000 epochs.

Overall, the code iteratively trains the model, updates the parameters, and tracks the training and test losses to monitor the model's performance during training.

```python
# Plot the loss curve
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('FFNN train set size 20 Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
```
This code plots the loss function for both train and test vs epoch.

```python
# De-normalize the predicted output
X_tensor = Variable(torch.Tensor(X).unsqueeze(1))
Y_pred1 = net(X_tensor.to(device)).cpu().detach().numpy() * Ystd + Ymean


# Plot the predicted curve with de-normalized data
plt.plot(X * Xstd + Xmean, Y * Ystd + Ymean, 'o', label='Data')
plt.plot(X * Xstd + Xmean, Y_pred1, label='Prediction')
plt.title('FFNN prediction on size 20 train set')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
```
Fit the model to the test data and plot the prediction function.

#### Part(iii) Repeat (ii) but use the first 10 and last 10 data points as training data. Then fit the model to the test data (which are the 10 held out middle data points). Compare these results to (ii)

```python
# Part (iii)
# Split the data into training and test sets
X_train, Y_train = X[:10], Y[:10]
X_test, Y_test = X[10:], Y[10:]
```
The rest is same to (ii)

#### Part(iv) Compare the models fit in homework one to the neural networks in (ii) and (iii)
See Computational Results.

### Problem 2
#### Part(i) Compute the first 20 PCA modes of the digit images.
Prepare dataset:
```python
# Part (i)
# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float32') / 255.
y = mnist.target.astype('int')

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Then we apply the 20 dimension PCA analysis:
```python
# Perform PCA analysis on the images
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Visualize the first 20 PCA modes
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(28, 28), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('PC %d' % (i+1))

# Add overall title
plt.suptitle('PCA Analysis: Visualization of the First 20 PCA Modes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```
This code performs PCA analysis on a dataset of images. It applies PCA to the training data and transforms it into the principal component space. Then, it applies the same transformation to the test data. The code visualizes the first 20 principal components as images in a grid of subplots. This visualization provides insights into the significant patterns or features captured by PCA.

#### Part (ii) Build a feed-forward neural network to classify the digits. Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers.

```python
# Convert data to PyTorch tensors
X_train_pca = torch.from_numpy(X_train_pca).float()
X_test_pca = torch.from_numpy(X_test_pca).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()
```

```python
# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.dropout(x, p=0.2)
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=0.2)
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.dropout(x, p=0.2)
        x = self.fc4(x)
        return nn.functional.log_softmax(x, dim=1)
```

This code defines a neural network architecture using the PyTorch framework. The neural network is implemented as a class called `Net`, which inherits from the `nn.Module` class.

The architecture consists of four fully connected layers (`fc1`, `fc2`, `fc3`, `fc4`). The input to the network is expected to have a size of 20, and the output size is set to 10, representing the number of classes in the classification problem.

In the constructor (`__init__` method) of the `Net` class, the fully connected layers are defined using the `nn.Linear` class. Each layer is specified with the input size and output size. For example, `self.fc1 = nn.Linear(20, 256)` creates a fully connected layer with 20 input neurons and 256 output neurons.

In the `forward` method, the input `x` passes through each layer sequentially. After each fully connected layer, a dropout operation is applied using `nn.functional.dropout` with a dropout probability of 0.2. Dropout helps prevent overfitting by randomly setting a fraction of the input units to 0 during training.

ReLU activation function (`nn.functional.relu`) is applied after `fc2` and `fc3` to introduce non-linearity into the network and enable it to learn complex patterns.

Finally, the output of the last fully connected layer (`fc4`) is returned after applying a log softmax function (`nn.functional.log_softmax`). The log softmax function converts the output values into log probabilities, which are suitable for multi-class classification problems.

Overall, this code defines a neural network with four fully connected layers, dropout regularization, and ReLU activation functions, suitable for performing classification tasks.


```python
# Initialize the neural network
model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the neural network
for epoch in tqdm(range(10000), desc='Training', unit='epoch', unit_scale=True, ncols=80, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
    optimizer.zero_grad()
    output = model(X_train_pca.to(device))
    loss = criterion(output, y_train.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 100 ==0:
      print(" Epoch: {:d}, Loss: {:.4f}".format(epoch+1, loss.item()))

# Evaluate the model on the test set
with torch.no_grad():
    output = model(X_test_pca.to(device))
    y_pred = torch.argmax(output, dim=1)
    accuracy = accuracy_score(y_test, y_pred.cpu())
    print("Accuracy: {:.2f}%".format(accuracy * 100))
```

This code initializes a neural network model with four fully connected layers, defines a cross-entropy loss function and an Adam optimizer, trains the model on a training dataset, and evaluates its accuracy on a test dataset. The training loop runs for 10,000 epochs, updating the model's parameters using backpropagation and optimizing with Adam. The model's performance is printed every 100 epochs. Finally, the trained model is used to make predictions on the test set, and the accuracy is calculated and displayed.



```python
# Reshape the data for the LSTM network
X_train_lstm = X_train.reshape((X_train.shape[0], 28, 28))
X_test_lstm = X_test.reshape((X_test.shape[0], 28, 28))

X_train_lstm = torch.tensor(X_train_lstm, dtype = torch.float32,  requires_grad = True)
X_test_lstm = torch.tensor(X_test_lstm, dtype = torch.float32,  requires_grad = True)
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(28, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return nn.functional.log_softmax(x, dim=1)

# Initialize the LSTM network
lstm_model = LSTMNet().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters())

# Train the LSTM network
for epoch in range(20):
    optimizer.zero_grad()
    output = lstm_model(X_train_lstm.to(device))
    loss = criterion(output, y_train.to(device))
    loss.backward()
    optimizer.step()
    print("Epoch: {:d}, Loss: {:.4f}".format(epoch+1, loss.item()))

# Evaluate the LSTM model on the test set
with torch.no_grad():
    output = lstm_model(X_test_lstm.to(device))
    y_pred = torch.argmax(output, dim=1)
    accuracy = accuracy_score(y_test, y_pred.cpu())
    print("LSTM Accuracy: {:.2f}%".format(accuracy * 100))
```
This code prepares and trains an LSTM network for a classification task. It reshapes the input data into sequences, initializes an LSTM network with an LSTM layer and a fully connected layer, defines the loss function and optimizer, and trains the network for a specified number of epochs. Finally, it evaluates the trained model on a test set and prints the accuracy.

SVM:
```python
# Train a SVM classifier
svm_model = SVC()
svm_model.fit(X_train_pca, y_train)
y_pred = svm_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy: {:.2f}%".format(accuracy * 100))
```

decision tree classifier:
```python
# Train a decision tree classifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_pca, y_train)
y_pred = tree_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy: {:.2f}%".format(accuracy * 100))
```

## Computational Results

### Problem 1(i)
<img width="558" alt="Screenshot 2023-05-12 at 3 34 49 AM" src="https://github.com/qchen4/EE399A/assets/98017700/77fa24dc-fec5-4881-bf69-d69b9795ce81">


Plot the loss curve


### Problem 1(ii)
<img width="578" alt="Screenshot 2023-05-12 at 3 35 34 AM" src="https://github.com/qchen4/EE399A/assets/98017700/44f96ad8-45a5-4ea0-bf64-6b0334043dde">




<img width="563" alt="Screenshot 2023-05-12 at 3 37 05 AM" src="https://github.com/qchen4/EE399A/assets/98017700/3b4cb5ba-3779-4ac0-bff8-555d74ff12ff">

### Problem 1(iii)
Training progress for FFNN train set size of 10:

<img width="600" alt="Screenshot 2023-05-12 at 3 43 03 AM" src="https://github.com/qchen4/EE399A/assets/98017700/29fa1264-d7ea-44ae-812b-20fc87268483">

<img width="581" alt="Screenshot 2023-05-12 at 3 38 01 AM" src="https://github.com/qchen4/EE399A/assets/98017700/0b61fd30-2f99-4e3e-a694-24a340d4a9f8">

<img width="566" alt="Screenshot 2023-05-12 at 3 38 11 AM" src="https://github.com/qchen4/EE399A/assets/98017700/c26e9453-c439-4ace-b68e-13afd4231046">


### Problem 1(iv)

<img width="726" alt="Screenshot 2023-05-12 at 3 39 10 AM" src="https://github.com/qchen4/EE399A/assets/98017700/b2feb515-0e31-49a9-827a-de7610fad32d">

Upon comparing the performance of the feedforward neural network (FFNN) with the least-square regression method, it becomes evident that the FFNN exhibits a significant issue of overfitting. It fails to provide meaningful predictions for future data points, indicating that the FFNN is simply attempting to fit the training dataset without effectively capturing useful features or patterns. The presence of overfitting suggests that the FFNN is excessively complex for the given training set, which consists of a relatively small number of data points (10 or 20). This highlights a fundamental difference between FFNN and regression approaches: FFNN requires a larger amount of training data to effectively learn and generalize patterns. The limited size of our training set impedes the FFNN's ability to generalize and extrapolate beyond the specific examples it has seen. To mitigate the overfitting issue and enhance the FFNN's performance, it would be beneficial to expand the training dataset with a more diverse and representative set of examples.




### Problem 2(i)

<img width="488" alt="Screenshot 2023-05-12 at 3 40 08 AM" src="https://github.com/qchen4/EE399A/assets/98017700/33809a24-5183-4622-8cd7-eccf1831854c">


<img width="1133" alt="Screenshot 2023-05-12 at 3 52 30 AM" src="https://github.com/qchen4/EE399A/assets/98017700/5e7ae7de-8b85-4532-81f9-2299a1dfaee9">


Accuracy of the FFNN: 96.36%







### Problem 2(ii)
LSVM training accuracies:

<img width="260" alt="Screenshot 2023-05-12 at 4 02 17 AM" src="https://github.com/qchen4/EE399A/assets/98017700/a13095ac-a05e-4bce-9a91-d5f1ead92a9f">



<img width="310" alt="Screenshot 2023-05-12 at 3 54 02 AM" src="https://github.com/qchen4/EE399A/assets/98017700/e5d79d70-98b2-4cb9-a86e-c910cb7f70c6">



<img width="384" alt="Screenshot 2023-05-12 at 3 54 11 AM" src="https://github.com/qchen4/EE399A/assets/98017700/c31b412f-b1f5-4336-8041-b87daf742f3a">





Here is a breif comparison of their accuracy on the testset using different classifiers and neural network.

Feedforward neural network: The accuracy of the three-layer feedforward neural network on the MNIST test set is 0.9729, which is a high accuracy compared to other classifiers.
* LSTM: The accuracy of the LSTM classifier on the MNIST test set is 24.71%, which is the lowest among all accuracies.
* SVM: The accuracy of the SVM classifier on the MNIST test set is 97.38%, which has the best performance among all four types of models.
* Decision tree: The accuracy of the decision tree classifier on the MNIST test set is 84.96%, which is lower than the accuracy of FFNN and SVM but higher than LSTM.
Therefore, the SVM has the highest accuracy among the tested classifiers. However, it is worth noting that the performance of these classifiers can vary depending on the hyperparameters and specific implementation used.


The LSTM network may perform poorly on the MNIST dataset due to several reasons:

1. **Limited sequential information**: The MNIST dataset consists of individual images of isolated digits, where the spatial arrangement of pixels does not carry significant sequential information. LSTMs are designed to capture long-term dependencies in sequential data, such as natural language processing or time series analysis. Since the MNIST dataset does not have strong temporal dependencies, LSTM may not effectively utilize its sequential modeling capabilities.

2. **Over-complexity**: LSTMs are relatively complex models with a large number of parameters. The simplicity of the MNIST dataset, where the task is to classify individual digits, does not require the level of complexity offered by an LSTM network. A simpler model, such as a feed-forward neural network or a convolutional neural network (CNN), is often sufficient for achieving good performance on the MNIST dataset.

3. **Limited dataset size**: The MNIST dataset consists of only 60,000 training images and 10,000 test images. LSTM models typically require a large amount of training data to generalize well and learn meaningful patterns. The limited size of the MNIST dataset may hinder the LSTM's ability to learn complex representations and lead to overfitting.

4. **Model architecture and hyperparameters**: The specific architecture and hyperparameters of the LSTM network can greatly impact its performance. If the LSTM network is not properly configured, it may struggle to learn meaningful representations from the MNIST data. Optimizing the architecture, hyperparameters, and regularization techniques can potentially improve the LSTM's performance on the dataset.

Considering these factors, it is often more effective to utilize simpler models like feed-forward neural networks or CNNs for the MNIST dataset, as they are better suited for image classification tasks and have demonstrated superior performance compared to LSTM models.

## Summary and Conclusions

In this assignment, we conducted two parts of analysis. In the first part, we used a three-layer feedforward neural network to fit a dataset consisting of 31 points. We observed that splitting the data into a training set of the first 10 points and a testing set of the last 10 points resulted in lower error compared to using the first 20 points for training and the remaining 10 points for testing.

Moving on to the second part, we performed analysis on the MNIST dataset, focusing on the first 20 principal components obtained through PCA. We constructed a feedforward neural network using these components and compared its performance with other classifiers such as SVM, LSTM, and decision trees. Among the classifiers, SVM and the feedforward neural network achieved relatively higher accuracy. Fine-tuning and optimizing the parameters of the feedforward neural network could potentially improve its performance further.

This assignment provided a valuable opportunity to apply neural networks and other classifiers to real-world datasets like MNIST. By exploring different models and evaluating their performance, we gained insights into choosing appropriate models. Additionally, utilizing PCA enhanced the performance of the classifiers. Overall, this assignment served as a practical exercise in neural networks and their applications, while also introducing us to other commonly used classifiers in machine learning.










