#include "MachineEstimate.h"
#include "CentralMachine.h"

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

    // Using Adam optimizer
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    arma::vec m(d * (p - d), arma::fill::zeros);
    arma::vec v(d * (p - d), arma::fill::zeros);
    arma::vec grad(d * (p - d), arma::fill::zeros);

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

            Y.subvec(startIdx, endIdx) -= arma::mean(Y.subvec(startIdx, endIdx));
            Y.subvec(startIdx, endIdx) /= arma::stddev(Y.subvec(startIdx, endIdx));

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

    for (int iter = 0; iter < maxIter; ++iter)
    {
        eScore.fill(0);

        if (useParallel)
        {
#pragma omp single
            {
                arma::mat eScoreLocalMat(machines, d * (p - d), arma::fill::zeros);
                arma::mat HLocal(d * (p - d), d * (p - d), arma::fill::zeros);
                {
                    int j = 0;
                    Rcpp::List localResult;
                    // omp_set_num_threads(12);
#pragma omp parallel for private(j) shared(machines) schedule(dynamic)
                    for (j = 0; j < machines; ++j)
                    {
                        eScoreLocalMat.row(j) = Rcpp::as<arma::vec>(estimateScore(X.rows(j * n, std::min((j + 1) * n, N) - 1),
                                                                                  beta, Y.subvec(j * n, std::min((j + 1) * n, N) - 1), h1, h2, h3)["eScore"])
                                                    .t();
                    }
                }

                // Combine results from all threads
                omp_set_num_threads(1);
                {
                    eScore = arma::sum(eScoreLocalMat, 0).t();
                    H += Rcpp::as<arma::mat>(estimateScore(X.rows(0, n), beta, Y.subvec(0, n), h1, h2, h3)["H"]);
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
                    H += Rcpp::as<arma::mat>(result["H"]);
            }
        }

        eScore /= machines;
        grad = arma::solve(H + 1e-5 * arma::eye(H.n_rows, H.n_cols), eScore);

        // Adam optimizer
        m = beta1 * m + (1 - beta1) * grad;
        v = beta2 * v + (1 - beta2) * grad % grad;

        betaNew.tail_rows(p - d) = beta.tail_rows(p - d) + 1e-3 * vecInv(m / (arma::sqrt(v) + epsilon), p, d).tail_rows(p - d);

        if (arma::norm(betaNew - beta, 2) < tol)
        {
            beta = betaNew;
            break;
        }
        beta = betaNew;
    }

    return beta;
}

// Sparse solution
arma::mat dmdrSparse(
    const arma::mat &data, int machines, int d,
    int maxIter, double tol, bool useParallel)
{
    int N = data.n_rows, p = data.n_cols - 1, n = N / machines;
    arma::vec Y = data.col(p);
    arma::mat X = data.cols(0, p - 1);
    // arma::arma_rng::set_seed(123); // 设置固定的种子
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

            Y.subvec(startIdx, endIdx) -= arma::mean(Y.subvec(startIdx, endIdx));
            Y.subvec(startIdx, endIdx) /= arma::stddev(Y.subvec(startIdx, endIdx));

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

    for (int iter = 0; iter < maxIter; ++iter)
    {
        eScore.fill(0);
        H.fill(0);
        H1.fill(0);
        z.fill(0);

        if (useParallel)
        {
#pragma omp single
            {
                arma::mat zLocalMat(machines, d * (p - d), arma::fill::zeros);
                arma::mat HaLocalMat(machines, d * (p - d), arma::fill::zeros);
                {
                    int j = 0;
                    // Rcpp::List localResult;
                    // omp_set_num_threads(12);
#pragma omp parallel for private(j) shared(machines) schedule(dynamic)
                    for (j = 0; j < machines; ++j)
                    {
                        Rcpp::List localResult = estimateScoreSparse(X.rows(j * n, std::min((j + 1) * n, N) - 1),
                                                                     beta, Y.subvec(j * n, std::min((j + 1) * n, N) - 1), h1, h2, h3);
                        zLocalMat.row(j) = Rcpp::as<arma::vec>(localResult["z"]).t();
                        HaLocalMat.row(j) = Rcpp::as<arma::vec>(localResult["HAlpha"]).t();
                    }
                }

                // Combine results from all threads
                omp_set_num_threads(1);
                {
                    z = arma::sum(zLocalMat, 0).t();
                    Ha = arma::sum(HaLocalMat, 0).t();
                    H1 = Rcpp::as<arma::mat>(estimateScoreSparse(X.rows(0, n), beta, Y.subvec(0, n), h1, h2, h3)["H"]);
                    Ha1 = HaLocalMat.row(0).t();
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

                result = estimateScoreSparse(xSub, beta, ySub, h1, h2, h3);

                z += Rcpp::as<arma::vec>(result["z"]);

                Ha += Rcpp::as<arma::vec>(result["HAlpha"]);

                if (j == 0)
                {
                    Ha1 = Rcpp::as<arma::mat>(result["HAlpha"]);
                    H1 = Rcpp::as<arma::mat>(result["H"]);
                }
            }
        }

        Ha /= machines;
        z /= machines;

        arma::vec betaCoef = Ha - Ha1 - z;

        // Solve 1/2 * x^T * H1 * x + x^T betaCoef + lambda * |x|_1
        // betaNew = vecInv(solveLasso(H1, betaCoef, 0.01), p, d);
        betaNew = vecInv(solveLasso(H1 + 1e-3 * arma::eye(d * (p - d), d * (p - d)), betaCoef, 1e-3 * std::sqrt(std::log(p) / n)), p, d);

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
