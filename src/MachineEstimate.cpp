#include "MachineEstimate.h"

// Epanichnikov kernel for one dimension
double epanKernel(double u)
{
    return (std::abs(u) <= 1) ? 0.75 * (1 - u * u) : 0.0;
}

// Multivariate Epanichnikov kernel
double epanKernelMulti(const arma::rowvec &u)
{
    double result = 1.0;
    for (const auto &e : u)
    {
        result *= epanKernel((e < 0.01) ? 0.01 : e);
    }
    return result < 0.01 ? 0.01 : result;
}

// Gaussian kernel
double gaussianKernel(const arma::rowvec &u)
{
    double result = 1.0;
    for (auto &e : u)
    {
        result *= std::exp(-0.5 * e * e);
    }

    return result < 0.01 ? 0.01 : result;
}

// Extract lower triangular block of a matrix
arma::vec vecLower(const arma::mat &m, int p, int d)
{
    return arma::vectorise(m.tail_rows(p - d));
}

// Inverse operation of vecLower
arma::mat vecInv(const arma::vec &v, int p, int d)
{
    arma::mat result(p, d, arma::fill::zeros);
    result.submat(0, 0, d - 1, d - 1) = arma::eye(d, d);
    result.tail_rows(p - d) = arma::reshape(v, p - d, d);
    return result;
}

arma::vec cmptWeights(const arma::mat &X, const arma::vec &Y, const arma::vec &mHat)
{
    /**
     * Compute weights based on kernel density estimation.
     *
     * This function calculates weights for each observation in the dataset using
     * a kernel-based approach. The weights are determined by the relationship between
     * the response variable `Y`, the kernel function applied to the predictors `X`,
     * and the provided estimated mean values `mHat`.
     *
     * @param X      An n x p matrix of predictors, where each row represents a data point in p-dimensional space.
     * @param Y      A vector of length n containing the response variable values.
     * @param mHat   A vector of length n representing the estimated mean values
     *               \(\widehat{m}(\mathbf{x}_i^{\mathrm{T}} \boldsymbol{\alpha})\) for each observation.
     *
     * @return       A vector of length n containing the computed weights for each observation.
     *               Each weight reflects the estimated variance of the residuals weighted by
     *               the kernel density around the corresponding observation.
     *
     * Explanation:
     * 1. For each observation \(k\), the function computes:
     *    - The numerator: a kernel-weighted sum of squared residuals,
     *      \( \sum_{i=1}^n K_h(\mathbf{x}_i - \mathbf{x}_k) \cdot (Y_i - \widehat{m}(\mathbf{x}_i^\mathrm{T} \boldsymbol{\alpha}))^2 \).
     *    - The denominator: a kernel-weighted sum of the kernel values, \( \sum_{i=1}^n K_h(\mathbf{x}_i - \mathbf{x}_k) \).
     * 2. The weight for observation \(k\) is given by the ratio of the numerator to the denominator.
     * 3. A Gaussian kernel function (`gaussianKernel`) is used for calculating the kernel value.
     *
     * Note:
     * - The kernel function is assumed to be Gaussian.
     * - The weights can be normalized later by dividing by their sum since it will not affect the results.
     */

    int n = X.n_rows;
    arma::vec weights(n, arma::fill::ones);

#pragma omp parallel for
    for (int k = 0; k < n; ++k)
    {
        double numerator = 0.0;
        double denominator = 0.0;

        for (int i = 0; i < n; ++i)
        {
            // double Kh = epanKernelMulti(X.row(i) - X.row(k));
            double Kh = gaussianKernel(X.row(i) - X.row(k));

            denominator += Kh;

            double residual = Y[i] - mHat[i];
            numerator += Kh * residual * residual;
        }
        weights[k] = numerator / denominator;
    }

    return weights / arma::sum(weights);
}

// Main algorithm implementation
Rcpp::List estimateScore(
    const arma::mat &X, const arma::mat &alpha, const arma::vec &Y,
    double h1, double h2, double h3)
{

    /**
     * @brief Estimate the score and related quantities for dimension reduction using kernel methods.
     *
     * This function computes the following estimates based on kernel density estimation:
     * - \(\hat{m}\): the local mean approximation of \(Y\) using a Taylor expansion.
     * - \(\hat{\mathbf{m}}_1\): the gradient of \(\hat{m}\) with respect to \(X^T \alpha\).
     * - \(\hat{E}\left\{w(x) \mid x^T \alpha\right\}\): the conditional expectation of weights \(w(x)\).
     * - \(\hat{E}\left\{x w(x) \mid x^T \alpha\right\}\): the conditional expectation of \(x w(x)\).
     * - \(\hat{S}\): the score matrix for subsequent optimization.
     * - Hessian matrix estimation for the score.
     *
     * @param X An \(n \times p\) matrix of predictors.
     * @param alpha A \(p \times d\) projection matrix for dimension reduction.
     * @param Y A vector of length \(n\) containing the response variable.
     * @param h1, h2, h3 Bandwidth parameters for kernel functions used in different computations.
     * @return A list containing:
     *         - \(\hat{m}\) (local mean estimates).
     *         - \(\hat{\mathbf{m}}_1\) (gradient estimates).
     *         - \(\hat{S}\) (score matrix).
     *         - Hessian matrix estimates (if applicable).
     */

    int n = X.n_rows, p = X.n_cols, d = alpha.n_cols;
    arma::mat XtA = X * alpha;
    arma::vec mHat(n, arma::fill::zeros);                  // $\hat{m}$
    arma::mat m1Hat(n, d, arma::fill::zeros);              // $\hat{m}_1 = \frac{\partial m(x^T \alpha)}{\partial x}$
    arma::vec eWHat(n, arma::fill::zeros);                 // $E w | x^T \alpha \in \mathbb R$
    arma::mat exWHat(n, p, arma::fill::zeros);             // $E x w | x^T \alpha \in \mathbb R^p$
    arma::mat scoreHat(n, d * (p - d), arma::fill::zeros); // $\hat{S}$
    arma::mat veclX(n, d * (p - d), arma::fill::zeros);    // vecl(x tilde)
    // arma::vec weights = cmptWeights(X, XtA, Y);

    for (int k = 0; k < n; ++k)
    {
        arma::vec K1(n, arma::fill::zeros); // , K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);
        arma::rowvec xkAlpha = XtA.row(k);

        for (int i = 0; i < n; ++i)
        {
            if (i != k)
            {
                arma::rowvec diff = XtA.row(i) - xkAlpha;
                K1(i) = gaussianKernel(diff / h1);
            }
        }

        arma::uvec idx = arma::find(arma::linspace(0, n - 1, n) != k);
        // arma::vec wSub = weights.elem(idx),
        arma::vec ySub = Y.elem(idx);
        arma::mat xSub = X.rows(idx);
        arma::vec k1Sub = K1.elem(idx); //, k2Sub = K2.elem(idx), k3Sub = K3.elem(idx);

        arma::mat xDiff = xSub.each_row() - X.row(k);
        arma::mat Z = join_horiz(arma::ones(idx.n_elem), xDiff * alpha);

        arma::vec coef = arma::solve(Z.t() * diagmat(k1Sub) * Z + 1e-5 * arma::eye(d + 1, d + 1),
                                     Z.t() * diagmat(k1Sub) * ySub);

        double b0 = coef(0);
        arma::vec b = coef.tail(d);

        // Estimate $\hat{m}$, $\hat{m}_1$
        mHat(k) = b0;
        m1Hat.row(k) = b.t();
    }

    // mHat.elem(arma::find_nonfinite(mHat)).zeros();
    mHat.elem(arma::find_nonfinite(mHat)).fill(arma::mean(Y));
    m1Hat.elem(arma::find_nonfinite(m1Hat)).zeros();

    arma::vec weights = cmptWeights(X, Y, mHat);

    for (int k = 0; k < n; ++k)
    {
        // Estimate $\hat{E} w$, $\hat{E} x w$, vecl(x tilde).

        arma::vec K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);

        for (int i = 0; i < n; ++i)
        {
            if (i != k)
            {
                arma::rowvec diff = XtA.row(i) - XtA.row(k);
                K2(i) = gaussianKernel(diff / h2);
                K3(i) = gaussianKernel(diff / h3);
            }
        }

        arma::uvec idx = arma::find(arma::linspace(0, n - 1, n) != k);
        arma::vec wSub = weights.elem(idx), ySub = Y.elem(idx);
        arma::mat xSub = X.rows(idx);
        arma::vec k2Sub = K2.elem(idx), k3Sub = K3.elem(idx);

        eWHat(k) = arma::accu(k2Sub % wSub) / arma::accu(k2Sub);

        exWHat.row(k) = arma::sum((k3Sub % wSub).t() * xSub, 0) / arma::accu(k3Sub);

        arma::rowvec xDiffCentral = X.row(k) - exWHat.row(k) / eWHat(k);
        // std::cout << "xDiffCentral = " << xDiffCentral << std::endl;
        veclX.row(k) = vecLower(xDiffCentral.t() * m1Hat.row(k), p, d).t();
        scoreHat.row(k) = (Y(k) - mHat(k)) * weights(k) * veclX.row(k);
    }

    scoreHat.elem(arma::find_nonfinite(scoreHat)).zeros();
    arma::vec eScore = arma::mean(scoreHat, 0).t();

    veclX.elem(arma::find_nonfinite(veclX)).zeros();
    arma::mat H = 1.0 / n * veclX.t() * arma::diagmat(weights) * veclX;

    return Rcpp::List::create(
        Rcpp::Named("mHat") = mHat,
        Rcpp::Named("m1Hat") = m1Hat,
        Rcpp::Named("eWHat") = eWHat,
        Rcpp::Named("exWHat") = exWHat,
        Rcpp::Named("scoreHat") = scoreHat,
        Rcpp::Named("eScore") = eScore,
        Rcpp::Named("H") = H);
}

// Estimate scores with sparse solutions
Rcpp::List estimateScoreSparse(
    const arma::mat &X, const arma::mat &alpha, const arma::vec &Y,
    double h1, double h2, double h3)
{
    // Compute Hessian matrix H times vecl(alpha) and z
    int n = X.n_rows, p = X.n_cols, d = alpha.n_cols;

    // Estimate Hessian H as in estimateScore
    arma::mat XtA = X * alpha;
    arma::vec mHat(n, arma::fill::zeros);
    arma::mat m1Hat(n, d, arma::fill::zeros);
    arma::vec eWHat(n, arma::fill::zeros);
    arma::mat exWHat(n, p, arma::fill::zeros);
    arma::mat scoreHat(n, d * (p - d), arma::fill::zeros);
    arma::mat veclX(n, d * (p - d), arma::fill::zeros);

    arma::vec zk(d * (p - d), arma::fill::zeros);
    arma::vec z(d * (p - d), arma::fill::zeros);

    // Compute weights as in estimateScore
    for (int k = 0; k < n; ++k)
    {
        arma::vec K1(n, arma::fill::zeros); //, K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);
        arma::rowvec xkAlpha = XtA.row(k);

        for (int i = 0; i < n; ++i)
        {
            if (i != k)
            {
                arma::rowvec diff = XtA.row(i) - xkAlpha;
                // K1(i) = epanKernelMulti(diff / h1);
                K1(i) = gaussianKernel(diff / h1);
                // K2(i) = epanKernelMulti(diff / h2);
                // K3(i) = epanKernelMulti(diff / h3);
            }
        }

        arma::uvec idx = arma::find(arma::linspace(0, n - 1, n) != k);
        arma::vec ySub = Y.elem(idx);
        arma::mat xSub = X.rows(idx);
        arma::vec k1Sub = K1.elem(idx); //, k2Sub = K2.elem(idx), k3Sub = K3.elem(idx);

        arma::mat xDiff = xSub.each_row() - X.row(k);
        arma::mat Z = join_horiz(arma::ones(idx.n_elem), xDiff * alpha);

        arma::vec coef = arma::solve(Z.t() * diagmat(k1Sub) * Z + 1e-5 * arma::eye(d + 1, d + 1),
                                     Z.t() * diagmat(k1Sub) * ySub);

        double b0 = coef(0);
        arma::vec b = coef.tail(d);

        // Estimate $\hat{m}$, $\hat{m}_1$
        mHat(k) = b0;
        m1Hat.row(k) = b.t();
    }

    // Repalce nan by the sample mean in mHat and m1Hat
    mHat.elem(arma::find_nonfinite(mHat)).fill(arma::mean(Y));
    m1Hat.elem(arma::find_nonfinite(m1Hat)).zeros();

    arma::vec weights = cmptWeights(X, Y, mHat);

    for (int k = 0; k < n; ++k)
    {
        // Estimate $\hat{E} w$, $\hat{E} x w$, vecl(x tilde).

        arma::vec K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);

        for (int i = 0; i < n; ++i)
        {
            if (i != k)
            {
                arma::rowvec diff = XtA.row(i) - XtA.row(k);

                K2(i) = gaussianKernel(diff / h2);
                K3(i) = gaussianKernel(diff / h3);
            }
        }

        arma::uvec idx = arma::find(arma::linspace(0, n - 1, n) != k);
        arma::vec wSub = weights.elem(idx), ySub = Y.elem(idx);
        arma::mat xSub = X.rows(idx);
        arma::vec k2Sub = K2.elem(idx), k3Sub = K3.elem(idx);

        eWHat(k) = arma::accu(k2Sub % wSub) / arma::accu(k2Sub);
        exWHat.row(k) = arma::sum((k3Sub % wSub).t() * xSub, 0) / arma::accu(k3Sub);
        arma::rowvec xDiffCentral = X.row(k) - exWHat.row(k) / eWHat(k);

        veclX.row(k) = vecLower(xDiffCentral.t() * m1Hat.row(k), p, d).t();
        scoreHat.row(k) = (Y(k) - mHat(k)) * weights(k) * veclX.row(k);

        arma::vec vecXDiff = vecLower(xDiffCentral.t() * m1Hat.row(k), p, d);
        zk = weights(k) * vecXDiff * (vecXDiff.t() * vecLower(alpha, p, d) + Y(k) - mHat(k));
        zk.elem(arma::find_nonfinite(zk)).zeros();
        z += zk;
    }

    scoreHat.elem(arma::find_nonfinite(scoreHat)).zeros();
    arma::vec eScore = arma::mean(scoreHat, 0).t();

    veclX.elem(arma::find_nonfinite(veclX)).zeros();
    arma::mat H = 1.0 / n * veclX.t() * arma::diagmat(weights) * veclX;

    return Rcpp::List::create(
        Rcpp::Named("H") = H,
        Rcpp::Named("HAlpha") = H * vecLower(alpha, p, d),
        Rcpp::Named("z") = z / n,
        Rcpp::Named("mHat") = mHat,
        Rcpp::Named("m1Hat") = m1Hat,
        Rcpp::Named("eWHat") = eWHat,
        Rcpp::Named("exWHat") = exWHat,
        Rcpp::Named("scoreHat") = scoreHat,
        Rcpp::Named("eScore") = eScore);
}

// // Estimate scores with sparse solutions
// Rcpp::List estimateScoreSparse(
//     const arma::mat &X, const arma::mat &alpha, const arma::vec &Y,
//     double h1, double h2, double h3)
// {
//     // Compute Hessian matrix H times vecl(alpha) and z

//     int n = X.n_rows, p = X.n_cols, d = alpha.n_cols;

//     // Estimate Hessian H as in estimateScore
//     arma::mat XtA = X * alpha;
//     arma::vec mHat(n, arma::fill::zeros);
//     arma::mat m1Hat(n, d, arma::fill::zeros);
//     arma::vec eWHat(n, arma::fill::zeros);
//     arma::mat exWHat(n, p, arma::fill::zeros);
//     arma::mat scoreHat(n, d * (p - d), arma::fill::zeros);
//     arma::mat veclX(n, d * (p - d), arma::fill::zeros);

//     arma::vec weights = cmptWeights(X, XtA, Y);

//     arma::vec z = arma::zeros(d * (p - d));
//     arma::vec zk = arma::zeros(d * (p - d));

//     for (int k = 0; k < n; ++k)
//     {
//         arma::vec K1(n, arma::fill::zeros), K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);
//         arma::rowvec xkAlpha = XtA.row(k);

//         for (int i = 0; i < n; ++i)
//         {
//             if (i != k)
//             {
//                 arma::rowvec diff = XtA.row(i) - xkAlpha;
//                 K1(i) = epanKernelMulti(diff / h1);
//                 K2(i) = epanKernelMulti(diff / h2);
//                 K3(i) = epanKernelMulti(diff / h3);
//             }
//         }

//         arma::uvec idx = arma::find(arma::linspace(0, n - 1, n) != k);
//         arma::vec wSub = weights.elem(idx), ySub = Y.elem(idx);
//         arma::mat xSub = X.rows(idx);
//         arma::vec k1Sub = K1.elem(idx), k2Sub = K2.elem(idx), k3Sub = K3.elem(idx);

//         arma::mat xDiff = xSub.each_row() - X.row(k);
//         arma::mat Z = join_horiz(arma::ones(idx.n_elem), xDiff * alpha);

//         std::cout << "     K1 = " << K1 << std::endl;
//         std::cout << "     Z = " << Z << std::endl;
//         std::cout << "     Z.t() * diagmat(k1Sub) * Z = " << Z.t() * diagmat(k1Sub) * Z << std::endl;

//         arma::vec coef = arma::solve(Z.t() * diagmat(k1Sub) * Z + 1e-5 * arma::eye(d + 1, d + 1),
//                                      Z.t() * diagmat(k1Sub) * ySub);

//         double b0 = coef(0);

//         std::cout << " b = " << coef.t() << std::endl;

//         arma::vec b = coef.tail(d);

//         mHat(k) = b0; // $\hat{m}$
//         m1Hat.row(k) = b.t();
//         eWHat(k) = arma::accu(k2Sub % wSub) / arma::accu(k2Sub);
//         exWHat.row(k) = arma::sum((k3Sub % wSub).t() * xSub, 0) / arma::accu(k2Sub);

//         arma::mat xDiffCentral = X.row(k) - exWHat.row(k) / eWHat(k);

//         // $\hat{x}_{k,j}$ defined in Eq. (3.6) and (2.2) and (4.8)
//         arma::vec vecXDiff = vecLower(xDiffCentral.t() * b.t(), p, d);

//         // Print quantity for debugging z
//         std::cout << "--------------------------------" << std::endl;
//         std::cout << "w(k) = " << weights(k) << std::endl;
//         std::cout << "vecXDiff = " << vecXDiff << std::endl;
//         std::cout << "vecLower(alpha, p, d) = " << vecLower(alpha, p, d) << std::endl;
//         std::cout << "Y(k) = " << Y(k) << std::endl;
//         std::cout << "mHat(k) = " << mHat(k) << std::endl;

//         vecXDiff.elem(arma::find_nonfinite(vecXDiff)).zeros();
//         zk = weights(k) * vecXDiff * (vecXDiff.t() * vecLower(alpha, p, d) + Y(k) - mHat(k));
//         zk.elem(arma::find_nonfinite(zk)).zeros();
//         z += zk;

//         scoreHat.row(k) = (Y(k) - mHat(k)) * weights(k) * vecXDiff.t();

//         veclX.row(k) = vecXDiff.t();
//     }

//     scoreHat.elem(arma::find_nonfinite(scoreHat)).zeros();
//     arma::vec eScore = arma::mean(scoreHat, 0).t();

//     veclX.elem(arma::find_nonfinite(veclX)).zeros();

//     arma::mat H = 1.0 / n * veclX.t() * arma::diagmat(weights) * veclX;
//     std::cout << " H = " << H << std::endl;
//     // H.elem(arma::find_nonfinite(H)).zeros();

//     return Rcpp::List::create(
//         Rcpp::Named("H") = H,
//         Rcpp::Named("HAlpha") = H * vecLower(alpha, p, d),
//         Rcpp::Named("z") = z / n,
//         Rcpp::Named("mHat") = mHat,
//         Rcpp::Named("m1Hat") = m1Hat,
//         Rcpp::Named("eWHat") = eWHat,
//         Rcpp::Named("exWHat") = exWHat,
//         Rcpp::Named("scoreHat") = scoreHat,
//         Rcpp::Named("eScore") = eScore);
// }
