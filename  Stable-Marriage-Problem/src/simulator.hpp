#ifndef PROPOSAL_H
#define PROPOSAL_H

#include "dense_matrix.hpp"

class Simulator {
  typedef la::dense_matrix::value_type value_type;

  // preferences of the proposer
  // - each row represents a proposer
  // - each column value is the index of an acceptor. The order of the indexes represents the preference
  //   of the given proposer for the acceptor. Leftmost indexes are the most preferred ones
  la::dense_matrix preferences_proposer;

  // preferences of the acceptor
  // - each row represents an acceptor
  // - each column value is the index of a proposer. The order of the indexes represents the preference
  //   of the given acceptor for the proposer. Leftmost indexes are the most preferred ones
  la::dense_matrix preferences_acceptor;

  // data structure that holds information about the matched couples. Each row represent an acceptor. Each
  // row has a single column that stores the index of the matched proposer
  // NOTE: we use the number of proposer to indicate that there is no match for the current acceptor
  la::dense_matrix matches;

  // data structure that holds information about the current status of the proposer. Each row represent a
  // proposer. Each row has a single column that stores the index of the next best acceptor that the
  // proposer might try to match with.
  // NOTE: we us the number of acceptor to indicate that we reached the bottom of the list
  la::dense_matrix proposer_status;

  // keep track of the number of proposer and acceptor
  value_type num_elements;

  // select the best proposer between the two candidates
  value_type select_best_proposer(const value_type acceptor_index,
                                  const value_type candidate_1,
                                  const value_type candidate_2) const;

  // find the matching proposer for the given acceptor index
  // NOTE: it will return an invalid index if the acceptor has no matches
  value_type get_matching_proposer(const value_type acceptor_index) const;

  // find the matching acceptor for the given proposer index
  // NOTE: it will return an invalid index if the proposer has no matches
  value_type get_matching_acceptor(const value_type proposer_index) const;

  // generate a matching matrix for the given proposal
  void update_matches(const la::dense_matrix& proposal);

  // generate the proposal matrix given the current matching situation
  la::dense_matrix compute_proposal();

public:
  // initializeSthe simulator parameteSs
  Simulator(const la::dense_matrix& proposer, const la::dense_matrix& acceptor);

  // solve the matchmaking problem, returning the best matches that we found in the problem
  la::dense_matrix run();
};

#endif // PROPOSAL_H
