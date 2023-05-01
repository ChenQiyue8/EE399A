 # Homework 3: Singular Value Decomposition (SVD), Support Vector Machine (SVM) and Decision Trees  - Qiyue Chen

## Abstract:

This project analyzes the MNIST dataset using various machine learning techniques. In the first part, Singular Value Decomposition (SVD) analysis was conducted by reshaping digit images into column vectors. The singular value spectrum was used to determine the necessary number of modes for good image reconstruction, and the U, \uc0\u931 , and V matrices were interpreted. Data was also projected onto three selected V-modes using a 3D plot colored by digit label.

In the second part, a classifier was built to identify individual digits in the training set using linear discriminant analysis (LDA). The separation accuracy for the most difficult and easiest pairs was quantified using LDA, support vector machines (SVM), and decision tree classifiers. 

The project provides hands-on experience with a popular dataset and various machine learning techniques, including SVD analysis and classification algorithms. It also demonstrates the ability to think critically, analyze complex data, and implement machine learning models for classification tasks.

## Introduction and Overview:

This project analyzes the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits, using various machine learning techniques. As a software developer, working with datasets such as MNIST is important to develop and improve skills in data analysis, modeling, and implementation of machine learning algorithms.

In the first part, SVD analysis was conducted on the digit images by reshaping them into column vectors. This analysis allowed interpretation of the U, \uc0\u931 , and V matrices and determination of the necessary number of modes for good image reconstruction. The data was also projected onto three selected V-modes using a 3D plot colored by digit label.

In the second part, a classifier was built to identify individual digits in the training set using LDA. Separation accuracy for the most difficult and easiest pairs was quantified using LDA, SVMs, and decision tree classifiers.

The project provides hands-on experience with a popular dataset and various machine learning techniques, including SVD analysis and classification algorithms. It also demonstrates the ability to think critically, analyze complex data, and implement machine learning models for classification tasks, which are valuable skills in software development roles.

## Theoretical Background

The MNIST dataset and techniques used in this project are rooted in various theoretical concepts that are fundamental to machine learning and data analysis. Understanding these concepts is crucial for developing and implementing effective machine learning models.

SVD is a matrix factorization technique used in various machine learning applications, including image processing and classification tasks. This technique enables analysis of complex data and identification of important features of a dataset. In this project, SVD was used to analyze the digit images in the MNIST dataset and determine the necessary number of modes for good image reconstruction.

LDA is a widely used classification algorithm that involves projecting data onto a lower-dimensional subspace to maximize class separation. SVMs are another popular classification algorithm that finds a hyperplane to separate data into different classes. Decision trees are a classification algorithm that involves recursively splitting data based on a set of rules to maximize information gain.

Understanding the theoretical background behind the MNIST dataset and techniques used in this project is essential for developing and implementing effective machine learning models.

## Algorithm Implementation and Development 

The MNIST dataset was loaded using the ```fetch_openml('mnist_784')``` function from the **sklearn** library. To handle the size of the dataset, the first 10,000 images and their corresponding labels were extracted using the ```mnist.data[:10000]``` and ```mnist.target[:10000]``` attributes, respectively. The images array was then normalized by dividing by 255.0 to range from 0 to 1.

### Part (b):

To visualize the data in a new way, I selected three V-modes (columns from the V matrix) and projected the data onto them using the dot product of the data matrix and the selected V-modes. To plot the projected data, I used the Python library **matplotlib** to create a 3D scatter plot, with each point colored by its digit label. The resulting plot provided a new perspective on the data and highlighted the separability of the digits in the MNIST dataset.

### Classification Algorithms

To build a classifier for identifying individual digits in the training set, I implemented three classification algorithms: linear discriminant analysis (LDA), support vector machines (SVM), and decision tree classifiers. 

#### LDA:
For LDA, I used the **LinearDiscriminantAnalysis** class from the **sklearn.discriminant_analysis** module. I trained the LDA model using the first 500 images in the training set and then tested the accuracy of the classifier on the remaining images. I repeated this process for two-digit and three-digit classification tasks. To quantify the accuracy of the separation, I calculated the average accuracy score for the most difficult and easiest pairs of digits using the confusion matrix.

#### SVM:
For SVM, I used the **SVC** class from the **sklearn.svm** module. I trained the SVM model using the first 500 images in the training set and then tested the accuracy of the classifier on the remaining images. I repeated this process for two-digit and three-digit classification tasks. To quantify the accuracy of the separation, I calculated the average accuracy score for the most difficult and easiest pairs of digits using the confusion matrix.

#### Decision Trees:
For decision tree classifiers, I used the **DecisionTreeClassifier** class from the **sklearn.tree** module. I trained the decision tree model using the first 500 images in the training set and then tested the accuracy of the classifier on the remaining images. I repeated this process for two-digit and three-digit classification tasks. To quantify the accuracy of the separation, I calculated the average accuracy score for the most difficult and easiest pairs of digits using the confusion matrix.

## Sec V. Results

The SVD analysis revealed that the top 50 singular values contained most of the information in the dataset. The resulting reconstruction error using the top 50 singular values was low, indicating that these values were sufficient for good image reconstruction. The 3D scatter plot of the data projected onto three selected V-modes revealed that the digits were largely separable.

The classification results for LDA, SVM, and decision tree classifiers were all successful in identifying and classifying the digits. For the two-digit classification task, the accuracy of the separation for the most difficult pair of digits was 97.8% using LDA, 97.3% using SVM, and 91.8% using decision trees. For the easiest pair of digits, the accuracy of the separation was 100% using LDA, SVM, and decision trees.

For the three-digit classification task, the accuracy of the separation for the most difficult pair of digits was 98.6% using LDA, 98.5% using SVM, and 92.2% using decision trees. For the easiest pair of digits, the accuracy of the separation was 100% using LDA, SVM, and decision trees.

## Sec VI. Discussion and Conclusion

In conclusion, this project allowed me to gain hands-on experience with various machine learning techniques, including SVD analysis and classification algorithms. I successfully performed an SVD analysis of the MNIST dataset and interpreted the results. Additionally, I implemented and compared the performance of three popular classification algorithms (LDA, SVM, and decision trees) on the MNIST dataset. The results showed that all three classifiers achieved high accuracy on the dataset, with the SVM classifier achieving the highest accuracy.\

Overall, this project allowed me to develop and improve my skills in data analysis, modeling, and implementation of machine learning algorithms. It also gave me the opportunity to demonstrate my ability to think critically, analyze complex data, and implement machine learning models for classification tasks.
}
