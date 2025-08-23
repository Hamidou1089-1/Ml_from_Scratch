//
// Created by hamid on 18/08/2025.
//

#pragma once

#include <Eigen/Dense>



class GradientDescent {
private:
double learning_rate_;

public:
    GradientDescent(double learning_rate);
    Eigen::VectorXd step(const Eigen::VectorXd& weights,const Eigen::VectorXd& gradient);
};


