#ifndef MACHINE_ESTIMATION_H
#define MACHINE_ESTIMATION_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' @name epanKernel
//' @title Epanichnikov Kernel for One Dimension
//' @description Computes the Epanichnikov kernel value for a scalar input.
//' @param u A numeric value representing the input.
//' @return A numeric value representing the kernel evaluation.
//' @export
double epanKernel(double u);

//' @name epanKernelMulti
//' @title Multivariate Epanichnikov Kernel
//' @description Computes the product of Epanichnikov kernel values for a multivariate input.
//' @param u A numeric row vector representing the input.
//' @return A numeric value representing the kernel evaluation.
//' @export
double epanKernelMulti(const arma::rowvec &u);

//' @name gaussianKernel
//' @title Gaussian Kernel
//' @description This function computes the Gaussian kernel value for a given input vector. The kernel is applied element-wise to the input vector `u` and returns the product of the Gaussian functions of each element.
//' @param u A numeric row vector (p) representing the input data. Each element of this vector is used to calculate the corresponding Gaussian kernel value.
//' @return A numeric value representing the result of the Gaussian kernel function applied to the input vector `u`. The value is calculated as the product of Gaussian functions applied element-wise to each component of `u`.
//'   If the result is less than 0.01, it returns 0.01 to avoid extremely small values that might cause numerical instability in further computations.
//' @export
double gaussianKernel(const arma::rowvec &u);

//' @name vecLower
//' @title Extract Lower Triangular Block of a Matrix
//' @description Extracts and vectorizes the lower triangular portion of a matrix.
//' @param m A numeric matrix input.
//' @param p Number of rows in the matrix.
//' @param d Number of columns in the triangular block.
//' @return A numeric vector representing the vectorized lower triangular block.
//' @export
arma::vec vecLower(const arma::mat &m, int p, int d);

//' @name vecInv
//' @title Inverse Operation of vecLower
//' @description Reconstructs a matrix from its vectorized lower triangular block.
//' @param v A numeric vector representing the lower triangular block.
//' @param p Number of rows in the reconstructed matrix.
//' @param d Number of columns in the triangular block.
//' @return A numeric matrix reconstructed from the input vector.
//' @export
arma::mat vecInv(const arma::vec &v, int p, int d);

//' @name cmptWeights
//' @title Compute Kernel-Based Weights
//' @description This function calculates observation weights using a kernel density estimation approach.
//'              The weights reflect the kernel-weighted variance of residuals for each observation.
//' @param X A numeric matrix (n x p) of predictors, where each row represents an observation.
//' @param Y A numeric vector (n) of response variable values.
//' @param mHat A numeric vector (n) of estimated mean values for each observation.
//' @return A numeric vector (n) of computed weights for each observation. Each weight is calculated as the
//'         ratio of the kernel-weighted sum of squared residuals to the kernel-weighted sum of kernel values.
//' @details The function uses the following formula to compute weights for each observation \(k\):
//' \deqn{
//'   \widehat{w}(\mathbf{x}_k) = \frac{\sum_{i=1}^n K_h(\mathbf{x}_i - \mathbf{x}_k) \cdot (Y_i - \widehat{m}(\mathbf{x}_i^\mathrm{T} \boldsymbol{\alpha}))^2}
//'   {\sum_{i=1}^n K_h(\mathbf{x}_i - \mathbf{x}_k)}
//' }
//' Here, \(K_h\) represents the kernel function, which is assumed to be Gaussian in the implementation.
//'
//' The computation is parallelized using OpenMP for improved performance on large datasets. Ensure your
//' environment supports OpenMP for optimal execution.
//'
//' @note The kernel function used is Gaussian (`gaussianKernel`) and the weights are normalized by dividing by their sum.
//' @export
// [[Rcpp::export]]
arma::vec cmptWeights(const arma::mat &X, const arma::vec &Y, const arma::vec &mHat);

//' @name estimateScore
//' @title Estimate Scores and Parameters
//' @description Implements the core algorithm for estimating scores and model parameters based on regression and kernel density estimation.
//' @param X A numeric matrix (n x p) of predictors, where each row corresponds to an observation and each column represents a predictor variable.
//' @param alpha A numeric matrix (p x d) of transformation coefficients used to project the predictor variables into a lower-dimensional space. The transformation is applied to each row of X.
//' @param Y A numeric vector (n) of responses corresponding to each observation in X. This is the dependent variable in the regression.
//' @param h1 A numeric value representing the bandwidth parameter for the regression kernel. This parameter controls the smoothing effect in the regression step, affecting how much weight is given to nearby observations during the local linear fitting.
//' @param h2 A numeric value representing the bandwidth parameter for weight estimation kernel. This bandwidth is used in estimating the weighted averages of the responses, controlling the smoothness of the estimated weights.
//' @param h3 A numeric value representing the bandwidth parameter for covariate adjustment kernel. It affects the kernel used to estimate the weighted averages of the predictors, determining how much influence nearby predictor values have on the estimation process.
//' @return A list containing the following components:
//'   - \code{mHat}: A numeric vector of estimated intercepts from the local linear regression at each observation. This represents the estimated value of the response variable after transformation using the projection matrix \code{alpha}.
//'   - \code{m1Hat}: A numeric matrix of estimated slope coefficients. Each column corresponds to the slope coefficients associated with each predictor variable after projection, representing how each predictor affects the response variable in the transformed space.
//'   - \code{eWHat}: A numeric vector of weighted averages of responses, calculated using kernel density estimation. This is the conditional expectation of the responses, weighted by the kernel function based on the given bandwidth \code{h2}.
//'   - \code{exWHat}: A numeric matrix of weighted averages of the predictor variables, similarly estimated using kernel density functions for each predictor. It represents the conditional expectation of the predictors, weighted by the kernel function based on the bandwidth \code{h3}.
//'   - \code{scoreHat}: A numeric matrix of estimated scores for each observation. These scores are computed based on the weighted residuals from the local regression, adjusted by the estimated weights from \code{eWHat} and \code{exWHat}.
//'   - \code{eScore}: A numeric vector of mean scores, representing the average score across all observations, useful for model evaluation or assessment of overall fit.
//'   - \code{H}: A numeric matrix representing the Hessian of the model at each observation, used to estimate second-order partial derivatives of the loss function. This matrix is important for assessing the curvature of the model and is essential for optimization steps involving second-order methods.
//' @export
// [[Rcpp::export]]
Rcpp::List estimateScore(
    const arma::mat &X, const arma::mat &alpha, const arma::vec &Y,
    double h1, double h2, double h3);

//' @name estimateScoreSparse
//' @title Estimate Sparse Scores and Parameters
//' @description Implements the core algorithm for estimating scores and parameters with an emphasis on sparsity. This method incorporates sparsity-inducing techniques into the regression and kernel density estimation process.
//'
//' @param X A numeric matrix (n x p) of predictors, where each row corresponds to an observation and each column represents a predictor variable.
//' @param alpha A numeric matrix (p x d) of transformation coefficients used to project the predictor variables into a lower-dimensional space, incorporating sparsity constraints. This matrix ensures that many coefficients may be zero, enhancing interpretability.
//' @param Y A numeric vector (n) of responses corresponding to each observation in X. This is the dependent variable in the regression.
//' @param h1 A numeric value representing the bandwidth parameter for the regression kernel. Controls the smoothing effect in the sparse regression step, determining the weight given to nearby observations.
//' @param h2 A numeric value representing the bandwidth parameter for weight estimation kernel. Determines the smoothness of the estimated weights, incorporating sparsity in weighted averages of responses.
//' @param h3 A numeric value representing the bandwidth parameter for covariate adjustment kernel. Controls the smoothness of the kernel used to estimate weighted averages of the predictors, ensuring sparsity in the estimation process.
//'
//' @return A list containing the following components:
//'   - \code{mHat}: A numeric vector of estimated intercepts from the sparse local regression at each observation. Represents the response variable after applying the sparse transformation matrix \code{alpha}.
//'   - \code{m1Hat}: A numeric matrix of estimated sparse slope coefficients. Each column corresponds to the slope coefficients associated with each predictor after projection, ensuring sparsity in the representation.
//'   - \code{eWHat}: A numeric vector of weighted averages of responses, calculated using sparse kernel density estimation. Represents the conditional expectation of the responses, weighted by the kernel function with sparsity regularization.
//'   - \code{exWHat}: A numeric matrix of weighted averages of the predictor variables, estimated using sparse kernel density functions. Represents the conditional expectation of predictors, incorporating sparsity constraints.
//'   - \code{scoreHat}: A numeric matrix of sparse scores for each observation, computed based on the residuals from sparse local regression, adjusted by the sparse weights from \code{eWHat} and \code{exWHat}.
//'   - \code{eScore}: A numeric vector of mean sparse scores, representing the average sparse score across observations. Useful for assessing overall model sparsity and fit.
//'   - \code{H}: A numeric matrix representing the sparse Hessian of the model at each observation. Captures the second-order partial derivatives of the loss function with sparsity constraints, essential for optimization steps involving second-order methods.
//'
//' @details
//' The `estimateScoreSparse` function builds upon the `estimateScore` function by incorporating sparsity-inducing techniques. These techniques enforce zero coefficients in the transformation matrix \code{alpha} and ensure sparse representations in all estimation steps. The sparse solution is particularly beneficial for high-dimensional data where many predictors may be irrelevant. 
//'
//' Sparsity enhances model interpretability and reduces computational complexity while preserving essential information. The algorithm iteratively refines estimates using sparse kernels and ensures convergence through second-order optimization methods. The Hessian matrix is adjusted to account for sparsity in curvature assessments.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List estimateScoreSparse(
    const arma::mat &X, const arma::mat &alpha, const arma::vec &Y,
    double h1, double h2, double h3);


// // [[Rcpp::export]]
// Rcpp::List estimateScoreSparse(
//     const arma::mat &X, const arma::mat &alpha, const arma::vec &Y,
//     double h1, double h2, double h3);

#endif // MACHINE_ESTIMATION_H
