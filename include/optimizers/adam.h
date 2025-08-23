//
// Created by hamid on 19/08/2025.
//

#pragma once

#include <Eigen/Dense>



class Adam {
private:
    double learning_rate_;
    double beta_1, beta_2;
    double corr_beta_1, corr_beta_2;
    Eigen::VectorXd m_, v_;
    Eigen::VectorXd m_hat_, v_hat_, denominator_;  

public:
    Adam(double learning_rate, double beta1, double beta2);

    Eigen::VectorXd step(const Eigen::VectorXd& weights, const Eigen::VectorXd& gradient);

};


