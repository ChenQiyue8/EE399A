# Analyzing Face Image Relationships through Correlation and Dimensionality Reduction Techniques on the Yalefaces Dataset
**Author**:

Qiyue Chen 

**Abstract**:

By utilizing techniques such as correlation matrices, eigenvectors, and Singular Value Decomposition (SVD), this study explores the connections among 2414 grayscale face images. The report analyzes image correlations and distinguishes highly correlated and uncorrelated pairs while computing principal component directions. The study compares eigenvectors and SVD modes and evaluates the variance captured by SVD modes. It also provides valuable insights into image relationships under various lighting conditions, showcasing the effectiveness of these methods in image analysis.

---

## Introduction

This report delves into the relationships between face images using the Yalefaces Dataset, which comprises 39 faces captured under 65 lighting scenes, resulting in a total of 2414 grayscale images downsized to 32x32 pixels. Various techniques, such as correlation matrix computation, eigenvector calculation, and Singular Value Decomposition (SVD) application, were explored to analyze and visualize the dataset.

Initially, a 100x100 correlation matrix was calculated for the first 100 images, and the results were displayed using pcolor. Highly correlated and uncorrelated image pairs were identified, and their visualization was carried out. Then, a specified set of images was chosen, and a 10x10 correlation matrix was computed, and the resulting matrix was visualized.

Additionally, the matrix `Y = XX^T` was created to determine the first six eigenvectors with the highest magnitude eigenvalue. The first six principal component directions were determined by performing SVD on matrix X. The norm of the difference between their absolute values was used to compare the first eigenvector and the first SVD mode.

Finally, the percentage of variance captured by each of the first six SVD modes was computed, and the corresponding modes were visualized. The study sheds light on face image relationships under different lighting conditions and validates the effectiveness of various techniques in analyzing and processing image data.

---

## Theoretical Background

In this section, we provide a brief theoretical background on the core concepts used in this study, namely correlation, eigenvectors, and Singular Value Decomposition (SVD).

#### Correlation

Correlation is a statistical measure that quantifies the degree to which two variables are related and indicates the strength and direction of their relationship. In the context of image analysis, correlation can be used to evaluate the similarity between two images by comparing their pixel intensities. A greater correlation coefficient indicates a stronger positive relationship.

#### Eigenvectors and Eigenvalues

Eigenvectors and eigenvalues are fundamental concepts in linear algebra and have important applications in various fields, including machine learning and data analysis. An eigenvector of a square matrix A is a non-zero vector v that satisfies the equation `Av = λv`, where λ is a scalar value known as the eigenvalue corresponding to the eigenvector v. In image analysis, eigenvectors and eigenvalues can be used for dimensionality reduction and to identify the principal components that capture the most variance in the data.

#### Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) is a powerful linear algebra technique that decomposes a matrix into the product of three matrices: U, S, and `V^T`. For a matrix A, the SVD can be represented as `A = UΣV^T`, where U and V are orthogonal matrices, and Σ is a diagonal matrix containing singular values in descending order. SVD has wide-ranging applications, including image processing, data compression, and dimensionality reduction. In this study, SVD is utilized to identify the principal component directions that capture the most variance in the Yalefaces dataset.

By understanding these fundamental concepts of correlation, eigenvectors, and Singular Value Decomposition, we can effectively apply them to analyze the relationships between images in the Yalefaces dataset and perform dimensionality reduction.

---


## Algorithm Implementation and Development

Initialization of Yalefaces Dataset
```
results = loadmat('./yalefaces.mat')
X = results['X']
```


### Problem (a): Computing and plotting the correlation matrix between the first 100 faces

Extracting the first 100 columns and computing the correlation matrix
```
X_100 = X[:, :100]

C = np.dot(X_100.T, X_100)
```

Plotting the correlation matrix using pcolor
```
plt.pcolor(C)
color_bar = plt.colorbar()
color_bar.set_label('Correlation Coefficient')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.title('Correlation Matrix (100x100)')
plt.show()
```


### Problem (b): Plotting most and least correlated faces


Find the two most highly correlated and most uncorrelated images
```
max_corr = np.max(C - np.eye(100)*C)
min_corr = np.min(C + 1e16 * np.eye(100))
most_corr = np.where(C == max_corr)
least_corr = np.where(C == min_corr)
```

Plot the two most highly correlated and most uncorrelated images
```
fig, axs = plt.subplots(1, 4)
axs[0].imshow(X[:, most_corr[0][0]].reshape(32, 32), cmap='gray')
axs[0].set_title('Image ' + str(most_corr[0][0] + 1))
axs[1].imshow(X[:, most_corr[1][0]].reshape(32, 32), cmap='gray')
axs[1].set_title('Image ' + str(most_corr[1][0] + 1))
axs[2].imshow(X[:, least_corr[0][0]].reshape(32, 32), cmap='gray')
axs[2].set_title('Image ' + str(least_corr[0][0] + 1))
axs[3].imshow(X[:, least_corr[1][0]].reshape(32, 32), cmap='gray')
axs[3].set_title('Image ' + str(least_corr[1][0] + 1))
plt.show()
```


### Problem (c): Computing and plotting the correlation matrix for specified images

These are the indices of the faces we want to find correlations
```
face_idx = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
faces_we_care = X[:, face_idx]
```

Extracting the specified images from dataset and computing the 10x10 correlation matrix
```
face_corr = np.dot(faces_we_care.T, faces_we_care)
```

Plotting the correlation matrix using pcolor
```
plt.pcolor(face_corr)
plt.colorbar()
plt.show()
```

### Problem (d): Creating the matrix `Y = XX^T` and finding the first 6 eigenvectors with the largest magnitude eigenvalues


Computing the matrix `Y = XX^T`
```
Y = np.dot(X, X.T)
```

Computing the eignevalues and eigenvectors of Y and gathering the first 6 eigenvectors with the largest magnitude eigenvalues
```
# Compute eigenvalues and eigenvectors of Y
eigenvalues, eigenvectors = np.linalg.eigh(Y)

# Sort eigenvalues and eigenvectors by descending order of eigenvalues
idx = np.argsort(eigenvalues)[::-1]
idx = np.argsort(np.abs(eigvals))[::-1][:6]
top_eigvecs = eigvecs[:, idx]
top_eigvals = eigvals[idx]
```

Print results
```
for i, v in enumerate(top_eigvecs.T):
    print("Eigenvector {} : {}; Eigenvalue is {}".format(i+1, v, top_eigvals[i]))
```

### Problem (e): Singular value decomposition of X and finding the first 6 principal component directions
```
U, S, Vt = np.linalg.svd(X)

for i in range(6):
    print('Principal component direction {} is: {}'.format(i+1, U[i]))
    print("Eigenvalue {} is: {}".format(i+1, S[i] **2))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(U[:,i].reshape(32,32), cmap= 'gray')
    plt.title(f'SVD mode {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### Problem (f): Computing the norm of the difference of absolute values of the first eigenvector `v_1` and the first SVD mode `u_1`

Gathering the first eigenvector `v_1` from the first six eigenvectors of `Y = XX^T` found earlier

Gathering the first SVD mode `u_1`

Computing the norm of the difference of the absolute values of `v_1` and `u_1`
```
norm = np.linalg.norm(np.abs(eigvec[:,0]) - np.abs(U[:,0]))
print("the norm of v1 and u1 is: {}".format(norm))
```


### Problem (g): Compute the percentage of variance captured by each of the first 6 SVD modes and plotting the first 6 SVD modes

Computing the percentage of variance captured by each of the first 6 SVD modes
```
total_var = np.sum(np.square(S))
pct_var_explained = (np.square(S) / total_var) * 100
pct_var_explained = np.round(pct_var_explained, 2)
```

Plotting the first 6 SVD modes
```
for i in range(6):
    print('Percentage of variance explained by principal component {} is: {}%'.format(i+1, pct_var_explained[i]))
```

---

## Computational Results

### Problem (a): Computing and plotting the correlation matrix between the first 100 faces

<p>
  <img src='https://github.com/qchen4/EE399A/blob/be54602053b45137fd7dff613586485f1c894d51/HW2/Figures/ProblemACorrelationMatrix.png'>
</p>

### Problem (b): Plotting most and least correlated faces

<p>
  <img src='https://github.com/qchen4/EE399A/blob/be54602053b45137fd7dff613586485f1c894d51/HW2/Figures/FaceCorrelation.png'>
</p>

### Problem (c): Computing and plotting the correlation matrix for specified images

<p>
  <img src='https://github.com/qchen4/EE399A/blob/be54602053b45137fd7dff613586485f1c894d51/HW2/Figures/ProblemCCorrelationMatrixSpecifiedImages.png'>
</p>

### Problem (f): Computing the norm of the difference of absolute values of the first eigenvector `v_1` and the first SVD mode `u_1`

    Norm of the difference of absolute values of v1 and u1:
    1.7409659223071522e-15

### Problem (g): Compute the percentage of variance captured by each of the first 6 SVD modes and plotting the first 6 SVD modes

<p>
  <img src='https://github.com/qchen4/EE399A/blob/be54602053b45137fd7dff613586485f1c894d51/HW2/Figures/SVDmodes.png'>
</p>

| SVD mode | Percentage of variance captured |
|---------:|:--------------------------------|
|        1 | 72.93%                          |
|        2 | 15.28%                          |
|        3 | 2.57%                           |
|        4 | 1.88%                           |
|        5 | 0.64%                           |
|        6 | 0.59%                           |

---

## Summary and Conclusions

This report presents an analysis of the Yalefaces dataset using techniques such as correlation matrices, eigenvectors, and Singular Value Decomposition (SVD) to investigate the relationships between face images under different lighting conditions.

We computed a 100x100 correlation matrix for the first 100 images in the dataset, identifying highly correlated and uncorrelated image pairs. Furthermore, we created a 10x10 correlation matrix for a specified set of images and visualized the resulting matrix.

Exploring dimensionality reduction techniques, we found the first six eigenvectors with the largest magnitude eigenvalue by creating the matrix `Y = XX^T`. We then performed SVD on matrix X to obtain the first six principal component directions. The comparison between the first eigenvector and the first SVD mode demonstrated similar results.

We computed the percentage of variance captured by each of the first six SVD modes, finding that the first mode captured 72.93% of the variance and the second mode captured 15.28%. Visualizing the first six SVD modes provided insight into the principal components of the dataset, demonstrating how SVD enables accurate feature space representation while reducing the dimensionality of the data, resulting in cheaper computation.

In conclusion, the analysis showcased the effectiveness of correlation, eigenvectors, and Singular Value Decomposition in understanding the relationships between images in the Yalefaces dataset and performing dimensionality reduction. These techniques can be applied to various image analysis and processing tasks, providing valuable insights into image relationships and enabling more efficient data representation.