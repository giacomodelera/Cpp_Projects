#include "simulator.hpp"

#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mpi.h>

Simulator::Simulator(const la::dense_matrix& proposer, const la::dense_matrix& acceptor)
    : preferences_proposer(proposer), preferences_acceptor(acceptor), num_elements(proposer.rows()) {
  matches         = la::dense_matrix(num_elements, 1, num_elements); // we start without matches
  proposer_status = la::dense_matrix(num_elements, 1, 0);            // we start with the best choice
}

la::dense_matrix::value_type Simulator::select_best_proposer(const value_type acceptor_index,
                                                             const value_type candidate_1,
                                                             const value_type candidate_2) const {
  for (value_type candidate_index = 0; candidate_index < num_elements; ++candidate_index) {
    const value_type proposer_index = preferences_acceptor(acceptor_index, candidate_index);
    if (proposer_index == candidate_1) {
      return candidate_1;
    } else if (proposer_index == candidate_2) {
      return candidate_2;
    }
  }
  return candidate_1; // both candidates are unknown, we pick the first one
}

la::dense_matrix::value_type Simulator::get_matching_proposer(const value_type acceptor_index) const {
  return matches(acceptor_index, 0);
}

la::dense_matrix::value_type Simulator::get_matching_acceptor(const value_type proposer_index) const {
  for (value_type acceptor_index = 0; acceptor_index < matches.rows(); ++acceptor_index) {
    if (matches(acceptor_index, 0) == proposer_index) {
      return acceptor_index;
    }
  }
  return matches.rows();
}

void Simulator::update_matches(const la::dense_matrix& proposal) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Determina il sottoinsieme di acceptor da processare per ogni processo
  const int acceptors_per_process = num_elements / size;
  const int extra = num_elements % size;
  const int start = rank * acceptors_per_process + std::min(rank, extra);
  const int end = start + acceptors_per_process + (rank < extra ? 1 : 0);

  // Crea una matrice locale per i match calcolati dal processo
  la::dense_matrix local_matches(num_elements, 1, num_elements); // Inizializza a num_elements (no match)

  // Loop sugli acceptor assegnati a questo processo
  for (value_type acceptor_index = start; acceptor_index < end; ++acceptor_index) {
    // Ottieni il proposer attualmente corrispondente all'acceptor
    const value_type previous_match = get_matching_proposer(acceptor_index);
    value_type best_proposer_index = previous_match;

    // Cerca tra i proposer chi ha fatto una proposta
    for (value_type proposer_index = 0; proposer_index < num_elements; ++proposer_index) {
      if (proposal(acceptor_index, proposer_index) == 1) { // Se c'è una proposta
        // Determina se è meglio del match precedente
        best_proposer_index = select_best_proposer(acceptor_index, best_proposer_index, proposer_index);
      }
    }

    // Memorizza il miglior proposer trovato
    local_matches(acceptor_index, 0) = best_proposer_index;

    // Log della scelta
    log_match(acceptor_index, best_proposer_index, previous_match);
  }

  // Combina i risultati di tutti i processi
  la::dense_matrix global_matches(num_elements, 1, num_elements); // Inizializza a num_elements (no match)
  MPI_Allgather(
    local_matches.data() + start,               // Dati locali da raccogliere
    end - start,                                // Numero di righe locali
    MPI_UNSIGNED,                               // Tipo dei dati
    global_matches.data(),                      // Buffer globale per raccogliere tutti i risultati
    end - start,                                // Numero di righe locali (uguale per ogni processo)
    MPI_UNSIGNED,                               // Tipo dei dati
    MPI_COMM_WORLD                              // Comunicatore
  );

  // Aggiorna la matrice globale dei match
  matches = global_matches;
}


la::dense_matrix Simulator::compute_proposal() {
  // La matrice di output, inizialmente zero-inizializzata.
  la::dense_matrix proposal(num_elements, num_elements, 0);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Divisione delle righe (proposers) tra i processi
  const int proposers_per_process = num_elements / size;
  const int extra = num_elements % size;

  // Calcolo del range di righe assegnate a questo processo
  const int start = rank * proposers_per_process + std::min(rank, extra);
  const int end = start + proposers_per_process + (rank < extra ? 1 : 0);

  // Loop locale sulle righe assegnate (proposers)
  for (value_type proposer_index = start; proposer_index < end; ++proposer_index) {
    // Ottieni l'attuale match per il proposer corrente
    const value_type current_match_index = get_matching_acceptor(proposer_index);

    if (current_match_index == num_elements) {
      // Il proposer non ha un match, propone al prossimo migliore
      const value_type next_best_index =
          preferences_proposer(proposer_index, proposer_status(proposer_index, 0));
      proposal(next_best_index, proposer_index) = 1;

      // Aggiorna lo stato del proposer
      proposer_status(proposer_index, 0) = std::min(num_elements - 1, proposer_status(proposer_index, 0) + 1);

      log_proposal(proposer_index, next_best_index); // Log della proposta
    } else {
      log_no_proposal(proposer_index, current_match_index); // Log del non-proposal
    }
  }

  // Raccolta dei risultati da tutti i processi
  la::dense_matrix global_proposal(num_elements, num_elements, 0);
  MPI_Allreduce(proposal.data(), global_proposal.data(), num_elements * num_elements, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Ritorno della matrice globale combinata
  return global_proposal;
}


la::dense_matrix Simulator::run() {
  bool is_stable = false;
  while (!is_stable) {
    // compute the next round of the match making
    const la::dense_matrix new_proposal = compute_proposal();
    update_matches(new_proposal);

    // if all the proposers have been matched, we found a stable match
    bool current_is_stable = true;
    for (value_type proposer_index = 0; proposer_index < matches.rows() && current_is_stable;
         ++proposer_index) {
      const value_type match_index = get_matching_acceptor(proposer_index);
      if (match_index == num_elements) {
        current_is_stable = false;
      }
    }

    // check if we need to replace the new matching with the previous one
    if (current_is_stable) {
      is_stable = true;
    }
  }

  return matches;
}
