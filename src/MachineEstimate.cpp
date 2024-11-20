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
    for (size_t j = 0; j < u.n_elem; ++j)
    {
        double e = std::abs(u[j]);
        e = e < 0.01 ? 0.1 : e;
        result *= epanKernel(u[j]);
    }
    return result;
}

double gaussianKernel(const arma::rowvec &u)
{
    double result = 1.0;
    for (auto &e : u)
    {
        result *= std::exp(-0.5 * e * e);
    }

    return result < 0.01 ? 0.1 : result;
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

// Compute weights
arma::vec cmptWeights(const arma::mat &X, const arma::vec &Y, const arma::vec &mHat)
{
    int n = X.n_rows;
    arma::vec weights(n, arma::fill::ones);

// Parallel
#pragma omp parallel for
    for (int k = 0; k < n; ++k)
    {
        double numerator = 0.0;
        double denominator = 0.0;

        for (int i = 0; i < n; ++i)
        {
            // 调用 epanKernelMulti 计算核值
            // double Kh = epanKernelMulti(X.row(i) - X.row(k));
            double Kh = gaussianKernel(X.row(i) - X.row(k));

            // 更新分母
            denominator += Kh;

            // 更新分子
            double residual = Y[i] - mHat[i];
            numerator += Kh * residual * residual;
        }

        // 计算最终权重
        weights[k] = numerator / denominator;
    }

    return weights / arma::sum(weights);
}

// Main algorithm implementation
Rcpp::List estimateScore(
    const arma::mat &X, const arma::mat &alpha, const arma::vec &Y,
    double h1, double h2, double h3)
{
    std::cout << "-------------------------" << std::endl;
    std::cout << "  X = " << X << std::endl;
    std::cout << "  Y = " << Y << std::endl;
    std::cout << "  alpha = " << alpha << std::endl;
    std::cout << "-------------------------" << std::endl;

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
        arma::vec K1(n, arma::fill::zeros), K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);
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

        // eWHat(k) = arma::accu(k2Sub % wSub) / arma::accu(k2Sub);
        // exWHat.row(k) = arma::sum((k3Sub % wSub).t() * xSub, 0) / arma::accu(k2Sub);

        // arma::mat xDiffCentral = X.row(k) - exWHat.row(k) / eWHat(k);
        // arma::vec vecXDiff = vecLower(xDiffCentral.t() * b.t(), p, d);

        // veclX.row(k) = vecXDiff.t();
    }

    // mHat.elem(arma::find_nonfinite(mHat)).zeros();
    mHat.elem(arma::find_nonfinite(mHat)).fill(arma::mean(Y));
    m1Hat.elem(arma::find_nonfinite(m1Hat)).zeros();
    // Repalce nan by the sample mean in mHat and m1Hat

    std::cout << " mHat = " << mHat << std::endl;
    std::cout << " Y = " << Y << std::endl;
    std::cout << " m1Hat = " << m1Hat << std::endl;

    arma::vec weights = cmptWeights(X, Y, mHat);

    std::cout << "weights = " << weights << std::endl;

    for (int k = 0; k < n; ++k)
    {
        // Estimate $\hat{E} w$, $\hat{E} x w$, vecl(x tilde).

        arma::vec K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);

        for (int i = 0; i < n; ++i)
        {
            if (i != k)
            {
                arma::rowvec diff = XtA.row(i) - XtA.row(k);
                // K2(i) = epanKernelMulti(diff / h2);
                // K3(i) = epanKernelMulti(diff / h3);

                // Repalce 0 by 0.1
                // K2(i) = K2(i) < 0.01 ? 0.1 : K2(i);
                // K3(i) = K3(i) == 0.01 ? 0.1 : K3(i);

                K2(i) = gaussianKernel(diff / h2);
                K3(i) = gaussianKernel(diff / h3);
            }
        }

        arma::uvec idx = arma::find(arma::linspace(0, n - 1, n) != k);
        arma::vec wSub = weights.elem(idx), ySub = Y.elem(idx);
        arma::mat xSub = X.rows(idx);
        arma::vec k2Sub = K2.elem(idx), k3Sub = K3.elem(idx);

        eWHat(k) = arma::accu(k2Sub % wSub) / arma::accu(k2Sub);
        // eWHat(k) = arma::is_finite(eWHat(k)) ? eWHat(k) : 1;

        exWHat.row(k) = arma::sum((k3Sub % wSub).t() * xSub, 0) / arma::accu(k3Sub);

        // std::cout << "  eWHat(k) = " << eWHat(k) << std::endl;
        // std::cout << "  exWHat.row(k) = " << exWHat.row(k) << std::endl;

        arma::rowvec xDiffCentral = X.row(k) - exWHat.row(k) / eWHat(k);
        // std::cout << "xDiffCentral = " << xDiffCentral << std::endl;
        veclX.row(k) = vecLower(xDiffCentral.t() * m1Hat.row(k), p, d).t();
        scoreHat.row(k) = (Y(k) - mHat(k)) * weights(k) * veclX.row(k);
    }

    std::cout << "  eWHat = " << eWHat << std::endl;
    std::cout << "  exWHat = " << exWHat << std::endl;
    std::cout << "  veclX = " << veclX << std::endl;
    std::cout << "  scoreHat = " << scoreHat << std::endl;

    scoreHat.elem(arma::find_nonfinite(scoreHat)).zeros();
    arma::vec eScore = arma::mean(scoreHat, 0).t();

    veclX.elem(arma::find_nonfinite(veclX)).zeros();
    arma::mat H = 1.0 / n * veclX.t() * arma::diagmat(weights) * veclX;
    // H.elem(arma::find_nonfinite(H)).zeros();

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

    arma::vec weights = cmptWeights(X, XtA, Y);

    arma::vec z = arma::zeros(d * (p - d));
    arma::vec zk = arma::zeros(d * (p - d));

    for (int k = 0; k < n; ++k)
    {
        arma::vec K1(n, arma::fill::zeros), K2(n, arma::fill::zeros), K3(n, arma::fill::zeros);
        arma::rowvec xkAlpha = XtA.row(k);

        for (int i = 0; i < n; ++i)
        {
            if (i != k)
            {
                arma::rowvec diff = XtA.row(i) - xkAlpha;
                K1(i) = epanKernelMulti(diff / h1);
                K2(i) = epanKernelMulti(diff / h2);
                K3(i) = epanKernelMulti(diff / h3);
            }
        }

        arma::uvec idx = arma::find(arma::linspace(0, n - 1, n) != k);
        arma::vec wSub = weights.elem(idx), ySub = Y.elem(idx);
        arma::mat xSub = X.rows(idx);
        arma::vec k1Sub = K1.elem(idx), k2Sub = K2.elem(idx), k3Sub = K3.elem(idx);

        arma::mat xDiff = xSub.each_row() - X.row(k);
        arma::mat Z = join_horiz(arma::ones(idx.n_elem), xDiff * alpha);

        std::cout << "     K1 = " << K1 << std::endl;
        std::cout << "     Z = " << Z << std::endl;
        std::cout << "     Z.t() * diagmat(k1Sub) * Z = " << Z.t() * diagmat(k1Sub) * Z << std::endl;

        arma::vec coef = arma::solve(Z.t() * diagmat(k1Sub) * Z + 1e-5 * arma::eye(d + 1, d + 1),
                                     Z.t() * diagmat(k1Sub) * ySub);

        double b0 = coef(0);

        std::cout << " b = " << coef.t() << std::endl;

        arma::vec b = coef.tail(d);

        mHat(k) = b0; // $\hat{m}$
        m1Hat.row(k) = b.t();
        eWHat(k) = arma::accu(k2Sub % wSub) / arma::accu(k2Sub);
        exWHat.row(k) = arma::sum((k3Sub % wSub).t() * xSub, 0) / arma::accu(k2Sub);

        arma::mat xDiffCentral = X.row(k) - exWHat.row(k) / eWHat(k);

        // $\hat{x}_{k,j}$ defined in Eq. (3.6) and (2.2) and (4.8)
        arma::vec vecXDiff = vecLower(xDiffCentral.t() * b.t(), p, d);

        // Print quantity for debugging z
        std::cout << "--------------------------------" << std::endl;
        std::cout << "w(k) = " << weights(k) << std::endl;
        std::cout << "vecXDiff = " << vecXDiff << std::endl;
        std::cout << "vecLower(alpha, p, d) = " << vecLower(alpha, p, d) << std::endl;
        std::cout << "Y(k) = " << Y(k) << std::endl;
        std::cout << "mHat(k) = " << mHat(k) << std::endl;

        vecXDiff.elem(arma::find_nonfinite(vecXDiff)).zeros();
        zk = weights(k) * vecXDiff * (vecXDiff.t() * vecLower(alpha, p, d) + Y(k) - mHat(k));
        zk.elem(arma::find_nonfinite(zk)).zeros();
        z += zk;

        scoreHat.row(k) = (Y(k) - mHat(k)) * weights(k) * vecXDiff.t();

        veclX.row(k) = vecXDiff.t();
    }

    scoreHat.elem(arma::find_nonfinite(scoreHat)).zeros();
    arma::vec eScore = arma::mean(scoreHat, 0).t();

    veclX.elem(arma::find_nonfinite(veclX)).zeros();

    arma::mat H = 1.0 / n * veclX.t() * arma::diagmat(weights) * veclX;
    std::cout << " H = " << H << std::endl;
    // H.elem(arma::find_nonfinite(H)).zeros();

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
