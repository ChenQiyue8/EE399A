# EE-399-HW4
**Author: Ben Li**
**Date: 5/7/2023**
**Course: SP 2023 EE399**

![nagesh-pca-1](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/Comp-1.gif)

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
