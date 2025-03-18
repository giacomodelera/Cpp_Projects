cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(assignment3 VERSION 1.0)
enable_language(CXX)

# Look for the MPI dependency
find_package(MPI REQUIRED C)

# define the application sources
set(header_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND header_files
  "${header_path}/dense_matrix.hpp"
  "${header_path}/logger.hpp"
  "${header_path}/simulator.hpp"
)
list(APPEND source_files
  "${source_path}/dense_matrix.cpp"
  "${source_path}/main.cpp"
  "${source_path}/logger.cpp"
  "${source_path}/simulator.cpp"
)

# define the compilation step
add_executable(main ${header_files} ${source_files})
target_include_directories(main PUBLIC "${header_path}")
set_target_properties(main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_compile_definitions(main PUBLIC "OMPI_SKIP_MPICXX") # OpenMPI
target_compile_definitions(main PUBLIC "MPICH_SKIP_MPICXX") # MPICH
target_link_libraries(main PUBLIC MPI::MPI_C)

