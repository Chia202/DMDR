#include "utils.h"

arma::mat projection(const arma::mat &B)
{
    arma::mat BtB_inv = arma::inv(arma::trans(B) * B + 1e-6 * arma::eye(B.n_cols, B.n_cols));
    arma::mat P = B * BtB_inv * arma::trans(B);
    return P;
}

double trCor(const arma::mat &beta, const arma::mat &beta_hat)
{
    arma::mat P_beta = projection(beta);
    arma::mat P_beta_hat = projection(beta_hat);
    double tr = arma::trace(P_beta_hat * P_beta);
    return tr / beta.n_cols;
}
