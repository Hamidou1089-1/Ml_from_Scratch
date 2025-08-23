//
// Created by hamid on 19/08/2025.
//

#pragma once

#include "loss_concept.h"
#include <Eigen/Dense>

template <LossConcept LossT>
class elastic_net {
private:
    double lambda1_;
    double lambda2_;
    LossT base_loss_;
public:
    elastic_net(double lambda1, double lambda2, LossT base_loss);

    double compute_loss(
       const Eigen::VectorXd& y_true,
       const Eigen::VectorXd& y_pred,
       const Eigen::VectorXd& current_weight
   );

    Eigen::VectorXd compute_gradient(
        const Eigen::VectorXd& y_true,
        const Eigen::VectorXd& y_pred,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& current_weight
    );
};
