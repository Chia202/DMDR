#ifndef CENTRAL_MACHINE_H
#define CENTRAL_MACHINE_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' @name dmdrDense
//' @title Distributed Mean Dimension Reduction with Dense Solutions
//' @description Implements distributed parameter estimation for dimension reduction using data partitioned across multiple machines.
//' The algorithm iteratively updates parameters by computing local scores and Hessian matrices, combining these to refine global estimates.
//'
//' @param data A numeric matrix where the last column represents the response variable, and preceding columns contain predictor variables.
//' @param machines An integer specifying the number of machines to distribute computation across.
//' @param d An integer indicating the target reduced dimension for parameter estimation.
//' @param maxIter An integer specifying the maximum number of iterations (default: 100).
//' @param tol A numeric value representing the convergence tolerance for parameter updates (default: 1e-6).
//' @param useParallel A logical flag indicating whether to use parallel computation (default: FALSE).
//'
//' @return A numeric matrix containing the final estimated reduced-rank coefficient matrix.
//'
//' @details
//' This function partitions the data across the specified number of machines, performs local computations to estimate
//' scores and Hessian matrices, and aggregates these to iteratively refine global parameter estimates. The bandwidth parameters
//' \eqn{h_1}, \eqn{h_2}, and \eqn{h_3} are dynamically adjusted based on the partition size \eqn{n} as \eqn{n^{-1 / (4 + d)}}.
//' The algorithm stops iterating when the maximum change in parameters falls below the specified `tol`.
//' Singular Value Decomposition (SVD) is applied to the final coefficient matrix to enhance stability and interpretability.
//'
//' @export
// [[Rcpp::export]]
arma::mat dmdrDense(
    const arma::mat &data, int machines, int d,
    int maxIter = 100, double tol = 1e-6, bool useParallel = false);


// //' @name dmdrDense
// //' @title Distributed Mean Dimension Reduction with Dense Solutions
// //' @description Performs distributed parameter estimation using data partitioned across multiple machines.
// //' Iteratively updates parameters by computing local scores and Hessian matrices, combining them to refine estimates.
// //'
// //' @param data A numeric matrix where the last column is the response variable and preceding columns are predictors.
// //' @param machines Integer specifying the number of machines for computation.
// //' @param d Integer indicating the reduced dimension for parameter estimation.
// //' @param maxIter Maximum number of iterations (default: 100).
// //' @param tol Convergence tolerance for parameter updates (default: 1e-6).
// //'
// //' @return A numeric matrix containing the final estimated reduced-rank coefficient matrix.
// //'
// //' @details
// //' The data is partitioned across `machines`, and local computations are performed to estimate scores and Hessians.
// //' These are aggregated to iteratively refine global parameter estimates. Bandwidths `h1`, `h2`, and `h3` are updated
// //' dynamically based on the partition size `n` as \eqn{n^{-1 / (4 + d)}}. The algorithm converges when parameter changes
// //' fall below `tol`. Singular value decomposition (SVD) is applied to the final matrix to ensure stability and interpretability.
// //'
// //' @export
// // [[Rcpp::export]]
// arma::mat dmdrDense(
//     const arma::mat &data, int machines, int d,
//     int maxIter = 100, double tol = 1e-6, bool useParallel = false);




//' @name dmdrSparse
//' @title Distributed Mean Dimension Reduction with Sparse Solutions
//' @description Performs distributed parameter estimation for dimension reduction while enforcing sparsity in the solution.
//' This method is designed for large-scale data partitioned across multiple machines and incorporates regularization for sparse outputs.
//'
//' @param data A numeric matrix where the last column represents the response variable, and preceding columns contain predictor variables.
//' @param machines An integer specifying the number of machines to distribute computation across.
//' @param d An integer indicating the target reduced dimension for parameter estimation.
//' @param maxIter An integer specifying the maximum number of iterations (default: 100).
//' @param tol A numeric value representing the convergence tolerance for parameter updates (default: 1e-6).
//' @param useParallel A logical flag indicating whether to use parallel computation (default: FALSE).
//'
//' @return A numeric matrix containing the final estimated sparse coefficient matrix.
//'
//' @details
//' The `dmdrSparse` function partitions the input data across the specified number of machines and performs local computations
//' to estimate sparse scores and Hessian matrices. Regularization is applied to enforce sparsity, making this approach suitable
//' for high-dimensional data where many predictor variables may not contribute significantly to the response.
//'
//' The algorithm iteratively aggregates local computations to refine global parameter estimates. Convergence is determined
//' when the maximum parameter update falls below the specified tolerance (`tol`). The final sparse solution enhances model
//' interpretability and reduces computational overhead. Parallel computation can be enabled with the `useParallel` flag for
//' faster execution on multi-core systems.
//'
//' @export
// [[Rcpp::export]]
arma::mat dmdrSparse(
    const arma::mat &data, int machines, int d,
    int maxIter = 100, double tol = 1e-6, bool useParallel = false);

//' @name solveLasso
//' @title Solve Lasso Regression Using Coordinate Descent
//' @description Implements the Lasso (Least Absolute Shrinkage and Selection Operator) regression using the coordinate descent algorithm. It solves for sparse regression coefficients by penalizing the \(\ell_1\)-norm of the coefficients.
//'
//' @param H1 A numeric matrix (p x p) representing the covariance matrix or a related positive-definite matrix derived from the predictors. A small diagonal perturbation is added to ensure numerical stability.
//' @param betaCoef A numeric vector (p) representing the linear coefficients or gradient vector, typically derived from the regression problem.
//' @param lambda A numeric value specifying the regularization parameter. Higher values of \code{lambda} enforce greater sparsity by shrinking more coefficients to zero (default: 0.1).
//' @param maxIter An integer specifying the maximum number of iterations for the coordinate descent algorithm (default: 100).
//' @param tol A numeric value specifying the convergence tolerance. The algorithm terminates when the change in coefficients between iterations falls below this threshold (default: 1e-6).
//'
//' @return A numeric vector (p) of estimated regression coefficients. Sparse solutions are achieved due to the \(\ell_1\)-penalty applied by the Lasso method.
//'
//' @details
//' The `solveLasso` function uses the coordinate descent algorithm to optimize the Lasso objective function:
//' \deqn{\frac{1}{2} \mathbf{x}^T H_1 \mathbf{x} + \mathbf{\beta}^T \mathbf{x} + \lambda \|\mathbf{x}\|_1,}
//' where \(\|\mathbf{x}\|_1\) is the \(\ell_1\)-norm of \(\mathbf{x}\), encouraging sparsity in the solution. 
//'
//' Each iteration updates one coefficient at a time while keeping others fixed. The update uses a soft-thresholding operator to shrink coefficients based on the penalty parameter \code{lambda}. The algorithm iteratively refines the coefficients until convergence is achieved or the maximum number of iterations is reached.
//'
//' To ensure numerical stability, a small diagonal perturbation (e.g., \(10^{-3}\)) is added to \code{H1}, forming the matrix \code{H} internally. This helps prevent singularity or poorly conditioned matrices from affecting convergence.
//'
//' @examples
//' \dontrun{
//' H1 <- matrix(c(4, 1, 1, 3), nrow = 2)
//' betaCoef <- c(-1, 2)
//' lambda <- 0.5
//' result <- solveLasso(H1, betaCoef, lambda)
//' print(result)  # Sparse solution with \ell_1 regularization
//' }
//' @export
// [[Rcpp::export]]
arma::vec solveLasso(const arma::mat& H1, const arma::vec& betaCoef, double lambda = 0.1, int maxIter = 100, double tol = 1e-6);

#endif // CENTRAL_MACHINE_H
