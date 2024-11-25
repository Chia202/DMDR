#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' @name projection
//' @title Compute Projection Matrix
//' @description Calculates the projection matrix for a given matrix \( B \), which represents the subspace defined by \( B \).
//'
//' @param B A numeric matrix (\( m \times n \)), representing the basis of a subspace.
//'
//' @return A numeric matrix (\( m \times m \)), which is the projection matrix \( P \):
//' \deqn{P = B (B^T B + \epsilon I)^{-1} B^T}
//' where \( \epsilon = 10^{-6} \) ensures numerical stability.
//'
//' @details
//' The projection matrix \( P \) is computed using a regularized inverse to ensure stability when \( B^T B \) is near-singular. 
//' This is useful for projecting vectors onto the subspace defined by the columns of \( B \).
//'
//' @examples
//' # Example in R
//' B <- matrix(rnorm(20), nrow = 5)
//' P <- projection(B)
//' print(P)
//'
//' @export
// [[Rcpp::export]]
arma::mat projection(const arma::mat &B);

//' @name trCor
//' @title Calculate Trace Correlation Between Subspaces
//' @description Computes the trace correlation between two subspaces defined by \( \beta \) and \( \beta_{\text{hat}} \), based on their projection matrices.
//'
//' @param beta A numeric matrix (\( m \times n \)), defining the first subspace.
//' @param beta_hat A numeric matrix (\( m \times p \)), defining the second subspace.
//'
//' @return A numeric scalar representing the trace correlation:
//' \deqn{\text{trCor} = \frac{\text{tr}(P_{\text{beta\_hat}} P_{\text{beta}})}{n}}
//' where \( P_{\text{beta}} \) and \( P_{\text{beta\_hat}} \) are the projection matrices for \( \beta \) and \( \beta_{\text{hat}} \), respectively, and \( n \) is the number of columns in \( \beta \).
//'
//' @details
//' Trace correlation measures the similarity between two subspaces by comparing their projection matrices. It ranges from 0 to 1, where higher values indicate greater similarity. The function internally uses the `projection` function to compute the required projection matrices.
//'
//' @examples
//' # Example in R
//' beta <- matrix(rnorm(20), nrow = 5)
//' beta_hat <- matrix(rnorm(15), nrow = 5)
//' trace_correlation <- trCor(beta, beta_hat)
//' print(trace_correlation)
//'
//' @export
// [[Rcpp::export]]
double trCor(const arma::mat &beta, const arma::mat &beta_hat);

#endif // UTILS_H
