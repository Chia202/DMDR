#ifndef MAVE_H
#define MAVE_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// 局部加权回归函数
arma::vec local_regression(const arma::mat& X, const arma::vec& Y, const arma::vec& z, double h);

// MAVE 的主要优化过程
// [[Rcpp::export]]
arma::mat mave_fit(const arma::mat& X, const arma::vec& Y, int d, double h = 1, int max_iter = 100, double tol = 1e-6);

#endif // MAVE_H
