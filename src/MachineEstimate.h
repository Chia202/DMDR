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
double epanKernelMulti(const arma::rowvec& u);

//' @name vecLower
//' @title Extract Lower Triangular Block of a Matrix
//' @description Extracts and vectorizes the lower triangular portion of a matrix.
//' @param m A numeric matrix input.
//' @param p Number of rows in the matrix.
//' @param d Number of columns in the triangular block.
//' @return A numeric vector representing the vectorized lower triangular block.
//' @export
arma::vec vecLower(const arma::mat& m, int p, int d);

//' @name vecInv
//' @title Inverse Operation of vecLower
//' @description Reconstructs a matrix from its vectorized lower triangular block.
//' @param v A numeric vector representing the lower triangular block.
//' @param p Number of rows in the reconstructed matrix.
//' @param d Number of columns in the triangular block.
//' @return A numeric matrix reconstructed from the input vector.
//' @export
arma::mat vecInv(const arma::vec& v, int p, int d);


// Compute Weights
arma::vec cmptWeights(const arma::mat& X, const arma::vec& Y, const arma::vec& mHat);

//' @name estimateScore
//' @title Estimate Scores and Parameters
//' @description Implements the main algorithm for estimating scores and parameters.
//' @param X A numeric matrix (n x p) of predictors.
//' @param alpha A numeric matrix (p x d) of transformation coefficients.
//' @param Y A numeric vector (n) of responses.
//' @param weights A numeric vector (n) of observation weights.
//' @param h1 Bandwidth parameter for regression kernel.
//' @param h2 Bandwidth parameter for weight estimation kernel.
//' @param h3 Bandwidth parameter for covariate adjustment kernel.
//' @param isCentral A boolean indicating whether to compute centralized scores.
//' @return A list containing:
//'   - \code{mHat}: Vector of estimated intercepts.
//'   - \code{m1Hat}: Matrix of estimated slope coefficients.
//'   - \code{eWHat}: Vector of weighted averages of responses.
//'   - \code{exWHat}: Matrix of weighted averages of predictors.
//'   - \code{scoreHat}: Matrix of estimated scores.
//'   - \code{eScore}: Vector of mean scores.
//'   - \code{H}: Matrix of estimated Hessians.
//' @export
// [[Rcpp::export]]
Rcpp::List estimateScore(
        const arma::mat& X, const arma::mat& alpha, const arma::vec& Y,
        double h1, double h2, double h3
);

// [[Rcpp::export]]
Rcpp::List estimateScoreSparse(
        const arma::mat& X, const arma::mat& alpha, const arma::vec& Y,
        double h1, double h2, double h3
);

#endif // MACHINE_ESTIMATION_H
 