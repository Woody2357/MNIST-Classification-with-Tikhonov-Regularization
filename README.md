# MNIST Classification with Tikhonov Regularization

This project focuses on classifying digits `1` and `7` from the MNIST dataset by finding a dividing surface that separates the two classes. The separating surface is tested on the corresponding digits from the test set.

---

## Problem Description

We aim to find a dividing quadratic hypersurface defined as:

$$ w^{\top} x + b = 0, $$

such that:

- \( w^{\top} x + b > 0 \) for (almost all) \( x_j \) corresponding to digit `1` (labeled with \( y_j = 1 \)).
- \( w^{\top} x + b < 0 \) for (almost all) \( x_j \) corresponding to digit `7` (labeled with \( y_j = -1 \)).

This task is modeled as a **nonlinear least squares problem**:

$$ f(w) = \frac{1}{2} \sum_{j=1}^n [r_j(w)]^2, \quad r_j(w) = \log\left(1 + e^{-q(x_j; w)}\right), $$

where:

$$ q(x_j; w) := y_j \left( x^{\top} W x + v^{\top} x + b \right). $$

---

## Implementations

### 1. Nonlinear Least Squares
- **`mnist_2categories_quadratic_NLLS_nPCA.m`**  
  Solves the nonlinear least squares problem to find a dividing quadratic hypersurface using the **Levenberg-Marquardt algorithm**. Additionally, it evaluates how the number of principal components (PCA) affects misclassification.

- **`mnist_2categories_quadratic_NLLS_GN.m`**  
  Solves the nonlinear least squares problem using the **Gauss-Newton algorithm**.

### 2. Tikhonov Regularization
The task is further extended to a **smooth optimization problem** with Tikhonov regularization:

$$ f(w) = \frac{1}{n} \sum_{j=1}^{n} \log\left(1 + e^{-q(x_j; w)}\right) + \frac{\lambda}{2} \|w\|^2, $$

where \( w \) is the \((d^2 + d + 1)\)-dimensional vector of coefficients \( \{W, v, b\} \).

- **`mnist_2categories_quadratic_NLLS_SGD.m`**  
  Solves this regularized problem using **Stochastic Gradient Descent (SGD)**. It includes experiments with:
  - Different batch sizes and step sizes.
  - Three step size decreasing strategies for convergence.

---

## Experiments

Experiments explore:
1. The effect of PCA on misclassification rates for quadratic hypersurfaces.
2. The impact of batch sizes, step sizes, and decreasing step size strategies on SGD performance.

---

## Reference

For more details, refer to the course webpage of [Professor Maria K. Cameron](https://www.math.umd.edu/~mariakc/AMSC660/scientificComputing1new.html).

---

### How to Run

Clone the repository and run the corresponding MATLAB scripts to reproduce the results. MATLAB version R2021a or later is recommended.
