#include "power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;


    double power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {

        // YOUR CODE GOES HERE

        // check norm of x0
        if (linear_algebra::norm(x0) != 1) {
            std::cerr << "ERROR: norm(x0) != 1" << std::endl;
            return NAN;
        }


        // initialize variables
        std::vector<double> x = x0;
        double nu_prev = 0;
        double nu;
        double residual = 1;
        double increment = 1;

        // while-loop
        std::size_t it = 0;
        while ( !converged(residual, increment) && it < max_it ){

            // compute z
            std::vector<double> z = A*x;

            // compute the eigenvalue
            nu = linear_algebra::scalar(x, z);

            // convergence criteria
            residual = linear_algebra::norm(z-nu*x);
            increment = abs(nu - nu_prev)/abs(nu);

            // next step
            nu_prev = nu;
            x = z;
            linear_algebra::normalize(x);
            ++it;
        }

        //std::cout << "iterations = " << it << std::endl;
        return nu;
    }



    bool power_iteration::converged(const double& residual, const double& increment) const
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