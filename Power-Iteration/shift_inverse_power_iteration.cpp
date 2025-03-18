#include "shift_inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double shift_inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const double& mu, const std::vector<double>& x0) const
    {

        // YOUR CODE GOES HERE

        // initialize variables
        linear_algebra::square_matrix A_tilde;

        // compute A_tilde = A - muI matrix
        A_tilde = A;
        for (size_t i = 0; i < A.size(); ++i) {
                A_tilde(i,i) -= mu;
        }

        // compute minimum eigenvalue of A_tilde using inverse_power_iteration
        double lambda_tilde = eigenvalue::inverse_power_iteration::solve(A_tilde, x0);

        // compute eigenvalue of A closest to mu
        return mu + lambda_tilde;
    }

} // eigenvalue