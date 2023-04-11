EE399A
=========

This is my project demostration for EE399A Spring 2023. 

.. contents:: Table of contents

HW1
---

Abstract
^^^^^^^^
This report presents a study that explores the use of non-linear regression to determine the parameters of a function f(x) = Acos(Bx) +Cx+D. The project involves generating a 2D loss landscape to visualize the relationship between the parameters and the resulting loss. Additionally, the first 20 data points are used as training data to fit a line, parabola, and 19th degree polynomial to the data, and their performance is evaluated on the test data

Sec. I. Introduction and Overview
^^^^^^^^^^^^
Model selection and parameter tuning are crucial steps in various fields that rely on fitting models to data, such as machine learning and data science. Non-linear regression is a popular technique used for fitting models to non-linear data. The goal of this study is to explore the use of non-linear regression to determine the parameters of a function f(x) = Acos(Bx) +Cx+D and to evaluate the performance of different models with varying complexities.

Sec. II. Theoretical Background
^^^^^^^^^^^^
Non-linear regression is a statistical technique used to fit non-linear functions to data. The least-squares method is commonly used to determine the best-fit parameters of a non-linear function. The method involves minimizing the sum of squared differences between the predicted values and the actual values. The loss function can be visualized as a loss landscape, where each point represents a set of parameter values and the loss value associated with those parameters. In addition, linear regression is a technique used to fit a linear equation to data. Polynomial regression is a generalization of linear regression that allows for fitting higher-order polynomials to data.

Sec. III. Algorithm Implementation and Development 
^^^^^^^^^^^^
The project involves implementing non-linear regression to determine the parameters of the function f(x) = Acos(Bx) +Cx+D. The least-squares method is used to minimize the sum of squared differences between the predicted values and the actual values. The algorithm involves an iterative process that updates the parameter values until the convergence criteria are met. The project also involves implementing linear regression and polynomial regression to fit different models to the data. The least-squares method is used to determine the best-fit parameters for each model.

Sec. IV. Computational Results
^^^^^^^^^^^^
The results of the project show that non-linear regression is an effective technique for fitting non-linear functions to data. The 2D loss landscape provides a visualization of the relationship between the parameters and the resulting loss. The results of the linear regression and polynomial regression show that the complexity of the model affects the performance on the test data. The 19th degree polynomial has the lowest training error, but it has the highest test error, indicating overfitting.

Sec. V. Summary and Conclusions
^^^^^^^^^^^^
In conclusion, this study highlights the importance of model selection and parameter tuning in various fields that rely on fitting models to data. Non-linear regression is an effective technique for fitting non-linear functions to data, and the 2D loss landscape provides a visualization of the relationship between the parameters and the resulting loss. The results of the linear regression and polynomial regression show that the complexity of the model affects the performance on the test data, and overfitting can occur if the model is too complex. Overall, this study provides valuable insights into the use of non-linear regression and different model complexities for fitting models to data.




