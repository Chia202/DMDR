#ifndef CENTRAL_MACHINE_H
#define CENTRAL_MACHINE_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' @name dmdrDense
//' @title Distributed Mean Dimension Reduction with Dense Solutions
//' @description Performs distributed parameter estimation using data partitioned across multiple machines.
//' Iteratively updates parameters by computing local scores and Hessian matrices, combining them to refine estimates.
//'
//' @param data A numeric matrix where the last column is the response variable and preceding columns are predictors.
//' @param machines Integer specifying the number of machines for computation.
//' @param d Integer indicating the reduced dimension for parameter estimation.
//' @param maxIter Maximum number of iterations (default: 1000).
//' @param tol Convergence tolerance for parameter updates (default: 1e-6).
//'
//' @return A numeric matrix containing the final estimated reduced-rank coefficient matrix.
//'
//' @details
//' The data is partitioned across `machines`, and local computations are performed to estimate scores and Hessians.
//' These are aggregated to iteratively refine global parameter estimates. Bandwidths `h1`, `h2`, and `h3` are updated
//' dynamically based on the partition size `n` as \eqn{n^{-1 / (4 + d)}}. The algorithm converges when parameter changes
//' fall below `tol`. Singular value decomposition (SVD) is applied to the final matrix to ensure stability and interpretability.
//'
//' @export
// [[Rcpp::export]]
arma::mat dmdrDense(
    const arma::mat &data, int machines, int d,
    int maxIter = 1000, double tol = 1e-6, bool useParallel = false);

// [[Rcpp::export]]
arma::mat dmdrSparse(
    const arma::mat &data, int machines, int d,
    int maxIter = 100, double tol = 1e-6, bool useParallel = false);

// [[Rcpp::export]]
arma::vec solveLasso(const arma::mat& H1, const arma::vec& betaCoef, double lambda = 0.1, int maxIter = 1000, double tol = 1e-6);

#endif // CENTRAL_MACHINE_H
