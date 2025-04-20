/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement non-blocking 1D communication scheme
//       along X axis.
//
// SUMMARY:
//     - 1D splitting along X
// NEW:
//     - >>> Non-blocking communications <<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "mpi.h"
#include "src/exercises.h"
#include "src/lbm_struct.h"
#include <stdbool.h>

/****************************************************/
// Initialize the communication structure for exercise 3
void lbm_comm_init_ex3(lbm_comm_t *comm, int total_width, int total_height) {
  // Reuse the initialization logic from exercise 1
  lbm_comm_init_ex1(comm, total_width, total_height);
}

/****************************************************/
// Perform ghost cell exchange using non-blocking communication
void lbm_comm_ghost_exchange_ex3(lbm_comm_t *comm, lbm_mesh_t *mesh) {
  // Determine if the current process is the first or last in the X dimension
  bool first = comm->rank_x == 0;
  bool last = comm->rank_x == comm->nb_x - 1;

  // Get the local mesh dimensions
  int height = comm->height;
  int width = comm->width;

  // Get the rank of the current process in the X dimension
  int rank = comm->rank_x;

  // Array to store MPI requests for non-blocking communication
  // 4 * height because:
  // - 2 * height for sends (left and right)
  // - 2 * height for receives (left and right)
  MPI_Request requests[4 * height];
  int req_idx = 0; // Index to track the current request

  // Set up non-blocking receives from the left neighbor
  if (!first) {
    for (int i = 0; i < height; i++) {
      MPI_Irecv(
          lbm_mesh_get_cell(mesh, 0, i), // Receive into the leftmost column
          DIRECTIONS,                    // Number of directions per cell
          MPI_DOUBLE,                    // Data type
          rank - 1,                      // Source rank (left neighbor)
          321,                           // Message tag
          MPI_COMM_WORLD,                // Communicator
          &requests[req_idx++]           // Store the request
      );
    }
  }

  // Set up non-blocking receives from the right neighbor
  if (!last) {
    for (int i = 0; i < height; i++) {
      MPI_Irecv(lbm_mesh_get_cell(mesh, width - 1,
                                  i), // Receive into the rightmost column
                DIRECTIONS, MPI_DOUBLE,
                rank + 1, // Source rank (right neighbor)
                123,      // Message tag
                MPI_COMM_WORLD, &requests[req_idx++]);
    }
  }

  // Set up non-blocking sends to the right neighbor
  if (!last) {
    for (int i = 0; i < height; i++) {
      MPI_Isend(lbm_mesh_get_cell(mesh, width - 2,
                                  i), // Send from the second-to-last column
                DIRECTIONS, MPI_DOUBLE,
                rank + 1, // Destination rank (right neighbor)
                321,      // Message tag
                MPI_COMM_WORLD, &requests[req_idx++]);
    }
  }

  // Set up non-blocking sends to the left neighbor
  if (!first) {
    for (int i = 0; i < height; i++) {
      MPI_Isend(lbm_mesh_get_cell(mesh, 1, i), // Send from the second column
                DIRECTIONS, MPI_DOUBLE,
                rank - 1, // Destination rank (left neighbor)
                123,      // Message tag
                MPI_COMM_WORLD, &requests[req_idx++]);
    }
  }

  // Wait for all non-blocking communications (sends and receives) to complete
  MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);
}