#include "inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {

        // YOUR CODE GOES HERE

        // check norm of x0
        if (linear_algebra::norm(x0) != 1) {
            std::cerr << "ERROR: norm(x0) != 1" << std::endl;
            return NAN;
        }

        // Invert A using LU-decomposition
        linear_algebra::square_matrix L, U;
        linear_algebra::lu(A,L,U);
        const std::size_t n = A.size();
        linear_algebra::square_matrix A_tilde(n);
        for (int i = 0; i < n; i++) {
            std::vector<double> b(n, 0.);
            b[i] = 1.;

            // Solve Ly = b (forward substitution)
            std::vector<double> y = linear_algebra::forwardsolve(L, b);

            // Solve Ux = y (backward substitution)
            std::vector<double> x = linear_algebra::backsolve(U, y);

            // Store the solution x in the inverse matrix
            for (int j = 0; j < n; j++) {
                A_tilde(j,i) = x[j];
            }
        }

        // initialize variables
        std::vector<double> x = x0;
        double lambda_prev = 0;
        double lambda;
        double residual = 1;
        double increment = 1;

        // while-loop
        std::size_t it = 0;
        while ( !converged(residual, increment) && it < max_it ) {

            // compute z
            std::vector<double> z = A_tilde*x;

            // compute the eigenvalue
            lambda = linear_algebra::scalar(x, z);

            // convergence criteria
            residual = linear_algebra::norm(z-(lambda*x));
            increment = abs(lambda - lambda_prev)/abs(lambda);

            // next step
            lambda_prev = lambda;
            x = z;
            linear_algebra::normalize(x);
            ++it;
        }

        //std::cout << "iterations = " << it << std::endl;
        return 1/lambda;
    }



    bool inverse_power_iteration::converged(const double& residual, const double& increment) const
    {
        bool conv;

        switch(termination) {
            case(RESIDUAL):
                conv = residual < tolerance;
                break;
            case(INCREMENT):
                conv = increment < tolerance;
                break;
            case(BOTH):
                conv = residual < tolerance && increment < tolerance;
                break;
            default:
                conv = false;
        }
        return conv;
    }

} // eigenvalue