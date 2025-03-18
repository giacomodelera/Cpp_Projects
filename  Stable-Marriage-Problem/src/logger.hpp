#ifndef LOGGER_H
#define LOGGER_H

#include "dense_matrix.hpp"

void log_match(const la::dense_matrix::value_type acceptor_index,
               const la::dense_matrix::value_type matched_proposer_index,
               const la::dense_matrix::value_type previous_proposer_index);

void log_proposal(const la::dense_matrix::value_type proposer_index,
                  const la::dense_matrix::value_type acceptor_index);

void log_no_proposal(const la::dense_matrix::value_type proposer_index,
                     const la::dense_matrix::value_type acceptor_index);

#endif // LOGGER_H
