# MNIST-Classification-with-Tikhonov-Regularization

The task is to select all images with digits 1 and all images with digits 7 from the training set, find a dividing surface that separates them, and test this dividing surface on the 1’s and 7’s from the test set.

We aim at finding a dividfing hyperplane $w^{\top}x+b=0$ with that $w^{\top}x+b>0$ for all (almost all) $x_j$ corresponding to 1 (labeled with $y_j=1$) and $w^{\top}x+b<0$ for all (almost all) $x_j$ corresponding to 7 (labeled with $y_j=-1$). This task can be modeled as a nonlinear least squares problem:
$$\begin{equation}f(w)=\frac{1}{2}\sum_{j=1}^n[r_j(w)]^2, \quad r_j(w)=\log\left(1+e^{-q(x_j;w)}\right),\end{equation}$$
where $q(x_j;w):=y_j(x^{\top}Wx+v^{\top}x+b)$.

‘mnist_2categories_quadratic_NLLS_nPCA.m’ solves this nonlinear least squares problem and finds a dividing quadratic
hypersurface using the Levenberg-Marquardt algorithm. It also finds out how the number of PCAs affects the number of misclassified digits.

'mnist_2categories_quadratic_NLLS_GN.m' solves this nonlinear least squares problem using the Gauss-Newton algorithm.

This task can be modeled as a smooth optimization problem with Tikhonov regularization:
$$ f(w)=\frac{1}{n}\sum_{j=1}^{n}\log\left(1+e^{-q(x_j;w)}\right)+\frac{\lambda}{2}\|w\|^2. $$
Here $w$ denotes the $(d^2+d+1)$-dimensional vector of coefficients of $\{W,v,b\}$.

‘mnist_2categories_quadratic_NLLS_SGD.m’ solves this regularized problem with Stochastic Gradient Descent. It also compare different batch sizes and step sizes. Also, it provides three stepsizes decreasing strategies.

For more details about the background, refer to [Professor Maria K. Cameron's webpage]([https://www.runoob.com](https://www.math.umd.edu/~mariakc/AMSC660/scientificComputing1new.html))
