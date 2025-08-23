//
// Created by hamid on 19/08/2025.
//



#include "linear_regression.h"
#include "gradient_descent.h"
#include "mean_squared_error.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;
int main() {
    LinearRegression<GradientDescent, MeanSquaredError> model(
        GradientDescent(0.01),
        MeanSquaredError(),
        100000
    );
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(10, 10);
    Eigen::VectorXd weight = Eigen::VectorXd::Random(10);
    Eigen::VectorXd y = X * weight;
    model.fit(X, y);
    Eigen::VectorXd y_pred = model.predict(X);



    cout << " y_True = "<< y << endl;
    cout << "Y_predicted = " << y_pred << endl;
    cout << "Score = " << 0.5*(y-y_pred).squaredNorm() / y.size() << endl;
    return 0;
}