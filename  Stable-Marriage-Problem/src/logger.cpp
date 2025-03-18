#include "logger.hpp"

#include <iostream>
#include <mpi.h>
#include <sstream>

void log_match(const la::dense_matrix::value_type acceptor_index,
               const la::dense_matrix::value_type matched_proposer_index,
               const la::dense_matrix::value_type previous_proposer_index) {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::ostringstream line;
  line << 'P' << world_rank << " - Acceptor " << acceptor_index << " has accepted " << matched_proposer_index
       << " (before was " << previous_proposer_index << ')' << std::endl;
  std::cout << line.str();
}

void log_proposal(const la::dense_matrix::value_type proposer_index,
                  const la::dense_matrix::value_type acceptor_index) {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::ostringstream line;
  line << 'P' << world_rank << " - Proposer " << proposer_index << " propose to " << acceptor_index
       << std::endl;
  std::cout << line.str();
}

void log_no_proposal(const la::dense_matrix::value_type proposer_index,
                     const la::dense_matrix::value_type acceptor_index) {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::ostringstream line;
  line << 'P' << world_rank << " - Proposer " << proposer_index << " is fine with " << acceptor_index
       << std::endl;
  std::cout << line.str();
}
