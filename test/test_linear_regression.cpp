//
// Created by hamid on 19/08/2025.
//



#include "linear_regression.h"
#include "gradient_descent.h"
#include "adam.h"
#include "mean_squared_error.h"
#include "elastic_net.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;
int main() {
    LinearRegression<Adam, MeanSquaredError> model(
        Adam(0.001, 0.9, 0.999),
         MeanSquaredError(),
        1000000, 
        1e-10,
        1000
    );
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(1000, 1000);
    Eigen::VectorXd weight = Eigen::VectorXd::Random(1000);
    Eigen::VectorXd y = X * weight;
    model.fit(X, y);
    Eigen::VectorXd y_pred = model.predict(X);

    cout << "Score = " << 0.5*(y-y_pred).squaredNorm() / y.size() << endl;
    return 0;
}