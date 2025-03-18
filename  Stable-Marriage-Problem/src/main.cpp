#include "dense_matrix.hpp"
#include "simulator.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
  // bail out if we don't have correct number of command line arguments
  if (argc < 3) {
    std::cerr << "Error: wrong number of parameters" << std::endl;
    std::cerr << std::endl;
    std::cerr << "USAGE: " << argv[0] << " ./input/apps.txt ./input/devices.txt" << std::endl;
    std::cerr << std::endl;
    return EXIT_FAILURE;
  }

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Figure out our place in the communicator
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Rank zero initialize the input matrix with preferences about devices and apps
  la::dense_matrix app_preferences, device_preferences;
  if (world_rank == 0) {
    std::ifstream app_reader(argv[1]), device_reader(argv[2]);
    app_preferences.read(app_reader);
    device_preferences.read(device_reader);
  }

  // broadcast the information about preferences to all the other proceses
  int num_elements = app_preferences.rows();
  MPI_Bcast(&num_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (world_rank > 0) {
    app_preferences    = la::dense_matrix(num_elements, num_elements);
    device_preferences = la::dense_matrix(num_elements, num_elements);
  }
  MPI_Bcast(app_preferences.data(), num_elements * num_elements, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(device_preferences.data(), num_elements * num_elements, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // declare the simulator of the marriage problem
  Simulator matchmaker = Simulator(app_preferences, device_preferences);

  // perform the actual simulation
  const la::dense_matrix perfect_match = matchmaker.run();

  // A correct MPI application always call the finalize function
  // NOTE: Always nice to see if something go wrong with MPI
  MPI_Finalize();
  return EXIT_SUCCESS;
}
