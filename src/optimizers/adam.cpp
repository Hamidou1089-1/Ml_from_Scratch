//
// Created by hamid on 19/08/2025.
//

#include "adam.h"



Adam::Adam(double learning_rate, double beta1, double beta2): learning_rate_(learning_rate), beta_1(beta1), beta_2(beta2) {
    corr_beta_1 = beta_1;
    corr_beta_2 = beta_2;
}

Eigen::VectorXd Adam::step(const Eigen::VectorXd& weight, const Eigen::VectorXd& gradient) {
     if (m_.size() == 0) {
        int dim = gradient.size();
        m_ = Eigen::VectorXd::Zero(dim);
        v_ = Eigen::VectorXd::Zero(dim);
        m_hat_ = Eigen::VectorXd::Zero(dim);
        v_hat_ = Eigen::VectorXd::Zero(dim);
        denominator_ = Eigen::VectorXd::Zero(dim);
     }
     
     m_ = beta_1*m_ + (1 - beta_1)*gradient;
     v_ = beta_2*v_ + (1 - beta_2)*gradient.array().square().matrix();

     double const1 = 1.0 / (1 - corr_beta_1);
     double const2 = 1.0 / (1 - corr_beta_2);

     m_hat_.noalias() = const1*m_;
     v_hat_.noalias() = const2*v_;

     corr_beta_1 *= beta_1;
     corr_beta_2 *= beta_2; 

     double eps = 1e-8;
     
     denominator_ = v_hat_.array().sqrt() + eps;


     return weight - learning_rate_*m_hat_.cwiseQuotient(denominator_);
}