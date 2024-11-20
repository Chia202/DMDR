#include "MachineEstimate.h"
#include "CentralMachine.h"
#include "Mave.h"

// Dense solution
arma::mat dmdrDense(
    const arma::mat &data, int machines, int d,
    int maxIter, double tol, bool useParallel)
{
    int N = data.n_rows, p = data.n_cols - 1, n = N / machines;
    arma::vec Y = data.col(p);
    arma::mat X = data.cols(0, p - 1);
    arma::mat beta = arma::zeros(p, d);
    beta.submat(0, 0, d - 1, d - 1) = arma::eye(d, d);
    double h1 = 1; // std::pow(n, -1.0 / (4 + d));
    double h2 = h1, h3 = h1;

    arma::mat betaNew = beta;
    arma::vec eScore(d * (p - d));
    arma::mat H(d * (p - d), d * (p - d));
    Rcpp::List result;

    // // Normalize the input in each machine
    // for (int j = 0; j < machines; ++j)
    // {
    //     int startIdx = j * n;
    //     int endIdx = std::min((j + 1) * n, N) - 1;
    //     arma::mat xSub = X.rows(startIdx, endIdx);
    //     arma::vec ySub = Y.subvec(startIdx, endIdx);

    //     arma::vec xMean = arma::mean(xSub, 0).t();
    //     arma::vec xStd = arma::stddev(xSub, 0, 0).t();
    //     xSub.each_row() -= xMean.t();
    //     xSub.each_row() /= xStd.t();

    //     X.rows(startIdx, endIdx) = xSub;

    //     // ySub -= arma::mean(ySub);
    //     // Y.subvec(startIdx, endIdx) = ySub / arma::stddev(ySub);
    // }

    // Normalize the input in each machine and whiten X
    if (useParallel)
    {
#pragma omp parallel for
        for (int j = 0; j < machines; ++j)
        {
            int startIdx = j * n;
            int endIdx = std::min((j + 1) * n, N) - 1;
            arma::mat xSub = X.rows(startIdx, endIdx);

            // Center the data (zero mean)
            arma::vec xMean = arma::mean(xSub, 0).t();
            xSub.each_row() -= xMean.t();

            // Compute covariance matrix
            arma::mat cov = arma::cov(xSub);

            // Perform Eigen decomposition of covariance matrix
            arma::vec eigval;
            arma::mat eigvec;
            arma::eig_sym(eigval, eigvec, cov);

            // Form the whitening transformation matrix
            arma::mat whiteningMatrix = eigvec * arma::diagmat(1.0 / arma::sqrt(eigval + 1e-6)) * eigvec.t();

            // Whiten the data
            xSub = xSub * whiteningMatrix.t();

// Update the original matrix
#pragma omp critical
            X.rows(startIdx, endIdx) = xSub;
        }
    }
    else
    {
        for (int j = 0; j < machines; ++j)
        {
            int startIdx = j * n;
            int endIdx = std::min((j + 1) * n, N) - 1;
            arma::mat xSub = X.rows(startIdx, endIdx);

            // Center the data (zero mean)
            arma::vec xMean = arma::mean(xSub, 0).t();
            xSub.each_row() -= xMean.t();

            // Compute covariance matrix
            arma::mat cov = arma::cov(xSub);

            // Perform Eigen decomposition of covariance matrix
            arma::vec eigval;
            arma::mat eigvec;
            arma::eig_sym(eigval, eigvec, cov);

            // Form the whitening transformation matrix
            arma::mat whiteningMatrix = eigvec * arma::diagmat(1.0 / arma::sqrt(eigval + 1e-6)) * eigvec.t();

            // Whiten the data
            xSub = xSub * whiteningMatrix.t();

            // Update the original matrix
            X.rows(startIdx, endIdx) = xSub;
        }
    }
    beta = mave_fit(X.rows(0, n), Y.subvec(0, n), d);

    std::cout << "  Estimated beta by MAVE = " << beta << std::endl;

    // beta.submat(0, 0, d - 1, d - 1) = arma::eye(d, d);
    // arma::mat Q, R; // 用于存储 QR 分解的结果
  
    // QR 分解 B11
    // arma::qr(Q, R, beta.rows(0, d-1));
    
    beta = beta * arma::inv(beta.rows(0, d-1) + 1e-3 * arma::eye(d, d));

    std::cout << "  Estimated beta = " << beta << std::endl;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        eScore.fill(0);
        // H.fill(0);

        // Uncomment the following block to parallelize the loop
        if (useParallel)
        {
#pragma omp parallel
            {
                // for (int j = 0; j < machines; ++j)
                // {
                //     int startIdx = j * n;
                //     int endIdx = std::min((j + 1) * n, N) - 1;
                //     arma::mat xSub = X.rows(startIdx, endIdx);
                //     arma::vec ySub = Y.subvec(startIdx, endIdx);

                //     result = estimateScore(xSub, beta, ySub, h1, h2, h3);

                //     eScore += Rcpp::as<arma::vec>(result["eScore"]);
                //     if (j == 0)
                //         H = Rcpp::as<arma::mat>(result["H"]);
                // }

                arma::vec eScoreLocal(d * (p - d), arma::fill::zeros);
                arma::mat HLocal(d * (p - d), d * (p - d), arma::fill::zeros);

#pragma omp for
                for (int j = 0; j < machines; ++j)
                {
                    int startIdx = j * n;
                    int endIdx = std::min((j + 1) * n, N) - 1;
                    arma::mat xSub = X.rows(startIdx, endIdx);
                    arma::vec ySub = Y.subvec(startIdx, endIdx);

                    Rcpp::List localResult = estimateScore(xSub, beta, ySub, h1, h2, h3);
                    eScoreLocal += Rcpp::as<arma::vec>(localResult["eScore"]);
                    if (j == 0) // Only one thread needs to initialize H
                    {
                        HLocal = Rcpp::as<arma::mat>(localResult["H"]);
                    }
                }

// Combine results from all threads
#pragma omp critical
                {
                    eScore += eScoreLocal;
                    H = HLocal;
                }
            }
        }
        else
        {
            for (int j = 0; j < machines; ++j)
            {
                int startIdx = j * n;
                int endIdx = std::min((j + 1) * n, N) - 1;
                arma::mat xSub = X.rows(startIdx, endIdx);
                arma::vec ySub = Y.subvec(startIdx, endIdx);

                result = estimateScore(xSub, beta, ySub, h1, h2, h3);

                eScore += Rcpp::as<arma::vec>(result["eScore"]);
                if (j == 0)
                    H = Rcpp::as<arma::mat>(result["H"]);
            }
        }

        eScore /= machines;
        betaNew.tail_rows(p - d) = beta.tail_rows(p - d) +
                                   1e-3 * vecInv(arma::solve(H + 1e-5 * arma::eye(H.n_rows, H.n_cols), eScore), p, d).tail_rows(p - d);

        if (arma::norm(betaNew - beta, 2) < tol)
        {
            beta = betaNew;
            break;
        }
        beta = betaNew;
    }

    return beta;

    // arma::mat U, V;
    // arma::vec s;
    // arma::svd(U, s, V, beta);

    // return U.cols(0, s.n_elem - 1) * V.t();
}

// Sparse solution
arma::mat dmdrSparse(
    const arma::mat &data, int machines, int d,
    int maxIter, double tol, bool useParallel)
{
    int N = data.n_rows, p = data.n_cols - 1, n = N / machines;
    arma::vec Y = data.col(p);
    arma::mat X = data.cols(0, p - 1);
    arma::arma_rng::set_seed(123); // 设置固定的种子
    arma::mat beta = arma::randn(p, d);
    beta.submat(0, 0, d - 1, d - 1) = arma::eye(d, d);
    double h1 = std::pow(n, -1.0 / (4 + d));
    double h2 = h1, h3 = h1;

    arma::mat betaNew = beta;
    arma::vec eScore(d * (p - d), arma::fill::zeros);
    arma::mat H(d * (p - d), d * (p - d), arma::fill::zeros);
    Rcpp::List result;

    arma::vec Ha(d * (p - d), arma::fill::zeros);
    arma::vec Ha1(d * (p - d), arma::fill::zeros);
    arma::mat H1(d * (p - d), d * (p - d), arma::fill::zeros);
    arma::vec z(d * (p - d), arma::fill::zeros);

    // Normalize the input in each machine
    for (int j = 0; j < machines; ++j)
    {
        int startIdx = j * n;
        int endIdx = std::min((j + 1) * n, N) - 1;
        arma::mat xSub = X.rows(startIdx, endIdx);
        arma::vec ySub = Y.subvec(startIdx, endIdx);

        arma::vec xMean = arma::mean(xSub, 0).t();
        arma::vec xStd = arma::stddev(xSub, 0, 0).t();
        xSub.each_row() -= xMean.t();
        xSub.each_row() /= xStd.t();

        X.rows(startIdx, endIdx) = xSub;

        ySub -= arma::mean(ySub);
        Y.subvec(startIdx, endIdx) = ySub / arma::stddev(ySub);
    }

    for (int iter = 0; iter < maxIter; ++iter)
    {
        eScore.fill(0);
        H.fill(0);
        H1.fill(0);
        z.fill(0);

        for (int j = 0; j < machines; ++j)
        {
            int startIdx = j * n;
            int endIdx = std::min((j + 1) * n, N) - 1;
            arma::mat xSub = X.rows(startIdx, endIdx);
            arma::vec ySub = Y.subvec(startIdx, endIdx);

            result = estimateScoreSparse(xSub, beta, ySub, h1, h2, h3);

            eScore += Rcpp::as<arma::vec>(result["eScore"]);
            H += Rcpp::as<arma::mat>(result["H"]);
            z += Rcpp::as<arma::vec>(result["z"]);

            Ha += Rcpp::as<arma::vec>(result["HAlpha"]);

            if (j == 0)
                Ha1 = Rcpp::as<arma::mat>(result["HAlpha"]);

            if (j == 0)
                H1 = Rcpp::as<arma::mat>(result["H"]);
        }

        eScore /= machines;
        // betaNew.tail_rows(p - d) = beta.tail_rows(p - d) +
        //    1e-5 * vecInv(arma::solve(H1 + 1e-5 * arma::eye(H.n_rows, H.n_cols), eScore), p, d).tail_rows(p - d);

        Ha /= machines;
        z /= machines;

        std::cout << "---------------------------------" << std::endl;
        std::cout << "Ha = " << Ha << std::endl;
        std::cout << "Ha1 = " << Ha1 << std::endl;
        std::cout << "z = " << z << std::endl;
        std::cout << "---------------------------------" << std::endl;

        arma::vec betaCoef = Ha - Ha1 - z;

        // Solve 1/2 * x^T * H1 * x + x^T betaCoef + lambda * |x|_1
        betaNew = vecInv(solveLasso(H1, betaCoef, 0.5), p, d);

        std::cout << "  betaNew = " << betaNew << std::endl;

        // std::cout << "Shape of H1 : " << H1.n_rows << " x " << H1.n_cols << std::endl;
        // std::cout << "Shape of betaCoef : " << betaCoef.n_rows << std::endl;

        if (arma::norm(betaNew - beta, 2) < tol)
        {
            beta = betaNew;
            break;
        }
        beta = betaNew;
    }

    return beta;
}

arma::vec solveLasso(const arma::mat &H1, const arma::vec &betaCoef, double lambda, int maxIter, double tol)
{

    std::cout << " I am solving Lasso " << std::endl;
    std::cout << " H1 = " << H1 << std::endl;
    std::cout << " betaCoef = " << betaCoef << std::endl;
    int p = H1.n_cols;
    arma::vec x(p, arma::fill::zeros);
    arma::vec x_old = x;

    arma::mat H = H1 + 1e-3 * arma::eye(p, p);

    for (int iter = 0; iter < maxIter; iter++)
    {
        for (int j = 0; j < p; j++)
        {
            // Compute the partial residual excluding the j-th variable
            double rj = dot(H.row(j), x) - H(j, j) * x(j);

            // Update x(j) using soft-thresholding
            double zj = -(rj + betaCoef(j)) / H(j, j);
            x(j) = std::max(0.0, zj - lambda / H(j, j)) - std::max(0.0, -zj - lambda / H(j, j));
        }

        // Check for convergence
        if (arma::norm(x - x_old, 2) < tol)
            break;
        x_old = x;
    }

    std::cout << " Lasso solution = " << x << std::endl;
    return x;
}

/*** R
# library(MASS)
#
# set.seed(123)
#
# p <- 16
# N <- 250
# m <- 5
# n <- N / m
# d <- 1
# beta <- 2 * c(1, 1, -1, 1, rep(0, p - 4))
#
# covMatrix <- outer(1:p, 1:p, function(i, j)
#     0.5 ^ abs(i - j))
#
# x <- mvrnorm(N, mu = rep(0, p), Sigma = covMatrix)
#
# meanFunc <- function(xBeta) {
#     sin(2 * xBeta) + 2 * exp(2 + xBeta)
# }
#
# varFunc <- function(xBeta) {
#     log(abs(2 + xBeta) + 1)
# }
#
# xBeta <- x %*% beta
# meanY <- meanFunc(xBeta)
# varY  <- varFunc(xBeta)
# y <- rnorm(N, mean = meanY, sd = sqrt(varY))
#
# data <- as.matrix(data.frame(x, y))
# colnames(data) <- c(paste0("x", 1:p), "y")
#
# # head(data)
#
# result <- distributedAlgorithm(data, m, d)
# print(result)
#
# sqrt(mean((result - beta) ^ 2))
*/
