// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// dmdrDense
arma::mat dmdrDense(const arma::mat& data, int machines, int d, int maxIter, double tol, bool useParallel);
RcppExport SEXP _DMDR_dmdrDense(SEXP dataSEXP, SEXP machinesSEXP, SEXP dSEXP, SEXP maxIterSEXP, SEXP tolSEXP, SEXP useParallelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type machines(machinesSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type useParallel(useParallelSEXP);
    rcpp_result_gen = Rcpp::wrap(dmdrDense(data, machines, d, maxIter, tol, useParallel));
    return rcpp_result_gen;
END_RCPP
}
// dmdrSparse
arma::mat dmdrSparse(const arma::mat& data, int machines, int d, int maxIter, double tol, bool useParallel);
RcppExport SEXP _DMDR_dmdrSparse(SEXP dataSEXP, SEXP machinesSEXP, SEXP dSEXP, SEXP maxIterSEXP, SEXP tolSEXP, SEXP useParallelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type machines(machinesSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type useParallel(useParallelSEXP);
    rcpp_result_gen = Rcpp::wrap(dmdrSparse(data, machines, d, maxIter, tol, useParallel));
    return rcpp_result_gen;
END_RCPP
}
// solveLasso
arma::vec solveLasso(const arma::mat& H1, const arma::vec& betaCoef, double lambda, int maxIter, double tol);
RcppExport SEXP _DMDR_solveLasso(SEXP H1SEXP, SEXP betaCoefSEXP, SEXP lambdaSEXP, SEXP maxIterSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type H1(H1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type betaCoef(betaCoefSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(solveLasso(H1, betaCoef, lambda, maxIter, tol));
    return rcpp_result_gen;
END_RCPP
}
// estimateScore
Rcpp::List estimateScore(const arma::mat& X, const arma::mat& alpha, const arma::vec& Y, double h1, double h2, double h3);
RcppExport SEXP _DMDR_estimateScore(SEXP XSEXP, SEXP alphaSEXP, SEXP YSEXP, SEXP h1SEXP, SEXP h2SEXP, SEXP h3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type h1(h1SEXP);
    Rcpp::traits::input_parameter< double >::type h2(h2SEXP);
    Rcpp::traits::input_parameter< double >::type h3(h3SEXP);
    rcpp_result_gen = Rcpp::wrap(estimateScore(X, alpha, Y, h1, h2, h3));
    return rcpp_result_gen;
END_RCPP
}
// estimateScoreSparse
Rcpp::List estimateScoreSparse(const arma::mat& X, const arma::mat& alpha, const arma::vec& Y, double h1, double h2, double h3);
RcppExport SEXP _DMDR_estimateScoreSparse(SEXP XSEXP, SEXP alphaSEXP, SEXP YSEXP, SEXP h1SEXP, SEXP h2SEXP, SEXP h3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type h1(h1SEXP);
    Rcpp::traits::input_parameter< double >::type h2(h2SEXP);
    Rcpp::traits::input_parameter< double >::type h3(h3SEXP);
    rcpp_result_gen = Rcpp::wrap(estimateScoreSparse(X, alpha, Y, h1, h2, h3));
    return rcpp_result_gen;
END_RCPP
}
// mave_fit
arma::mat mave_fit(const arma::mat& X, const arma::vec& Y, int d, double h, int max_iter, double tol);
RcppExport SEXP _DMDR_mave_fit(SEXP XSEXP, SEXP YSEXP, SEXP dSEXP, SEXP hSEXP, SEXP max_iterSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(mave_fit(X, Y, d, h, max_iter, tol));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_DMDR_dmdrDense", (DL_FUNC) &_DMDR_dmdrDense, 6},
    {"_DMDR_dmdrSparse", (DL_FUNC) &_DMDR_dmdrSparse, 6},
    {"_DMDR_solveLasso", (DL_FUNC) &_DMDR_solveLasso, 5},
    {"_DMDR_estimateScore", (DL_FUNC) &_DMDR_estimateScore, 6},
    {"_DMDR_estimateScoreSparse", (DL_FUNC) &_DMDR_estimateScoreSparse, 6},
    {"_DMDR_mave_fit", (DL_FUNC) &_DMDR_mave_fit, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_DMDR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
