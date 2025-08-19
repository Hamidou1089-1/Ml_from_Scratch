//
// Created by hamid on 10/08/2025.
//

#include "LinearRegression.h"



LinearRegression::LinearRegression(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction> loss, double learningRate, long iterations):
    _optimizer(std::move(optimizer)), _loss(std::move(loss)) {
this->iterations = iterations;
}