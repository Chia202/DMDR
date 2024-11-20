#include "Mave.h"

// 局部加权回归函数
arma::vec local_regression(const arma::mat& X, const arma::vec& Y, const arma::vec& z, double h) {
    int n = X.n_rows;
    arma::vec weights(n);
    
    for (int i = 0; i < n; i++) {
        weights(i) = exp(-arma::norm(X.row(i) - z.t(), 2) / (2 * h * h));
    }
    
    arma::mat W = arma::diagmat(weights);
    arma::mat XtW = X.t() * W;
    arma::mat beta = arma::solve(XtW * X, XtW * Y); // 最小二乘解
    
    return beta;
}

// MAVE 的主要优化过程
arma::mat mave_fit(const arma::mat& X, const arma::vec& Y, int d, double h, int max_iter, double tol) {
    int p = X.n_cols;
    arma::mat B = arma::randn(p, d); // 初始化投影矩阵
    B = arma::orth(B); // 正交化
    arma::mat B_old;
    
    for (int iter = 0; iter < max_iter; iter++) {
        B_old = B;
        
        // Step 1: 计算投影后的数据
        arma::mat Z = X * B; // 投影到低维空间
        
        // Step 2: 局部加权回归估计
        arma::mat new_B(p, d, arma::fill::zeros);
        for (int i = 0; i < X.n_rows; i++) {
            arma::vec beta = local_regression(Z, Y, Z.row(i).t(), h);
            new_B += (X.row(i).t() * beta.t());
        }
        
        // Step 3: 更新投影矩阵
        B = arma::orth(new_B);
        
        // 检查收敛
        if (arma::norm(B - B_old, "fro") < tol) {
            break;
        }
    }
    
    return B;
}
