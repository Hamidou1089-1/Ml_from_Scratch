//
// Created by hamid on 19/08/2025.
//

#pragma once

#include <iostream>
#include <concepts>
#include <Eigen/Dense>


template <typename T>
concept LossConcept = requires(T loss, const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred, const Eigen::MatrixXd& X, const Eigen::VectorXd& current_weight) {
 {loss.compute_loss(y_true, y_pred, current_weight)} -> std::convertible_to<double>;
 {loss.compute_gradient(y_true, y_pred, X, current_weight)} -> std::convertible_to<Eigen::VectorXd>;
};