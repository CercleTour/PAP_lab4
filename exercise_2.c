/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement odd/even 1D blocking communication scheme
//       along X axis.
//
// SUMMARY:
//     - 1D splitting along X
//     - Blocking communications
// NEW:
//     - >>> Odd/even communication ordering <<<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "mpi.h"
#include "src/exercises.h"
#include "src/lbm_struct.h"
#include <stdbool.h>

/****************************************************/
// Initialize the communication structure for exercise 2
void lbm_comm_init_ex2(lbm_comm_t *comm, int total_width, int total_height) {
  // Reuse the initialization logic from exercise 1
  lbm_comm_init_ex1(comm, total_width, total_height);
}

/****************************************************/
// Perform ghost cell exchange using odd/even blocking communication
void lbm_comm_ghost_exchange_ex2(lbm_comm_t *comm, lbm_mesh_t *mesh) {
  // Determine if the current process is the first or last in the X dimension
  bool first = comm->rank_x == 0;
  bool last = comm->rank_x == comm->nb_x - 1;

  // Get the local mesh dimensions
  int height = comm->height;
  int width = comm->width;

  // Get the rank of the current process in the X dimension
  int rank = comm->rank_x;

  // Odd ranks send first, then receive
  if (rank % 2 != 0) { // ODD: Send before receiving
    // Send to the right neighbor
    if (!last) {
      for (int i = 0; i < height; i++) {
        MPI_Send(lbm_mesh_get_cell(mesh, width - 2,
                                   i), // Send from the second-to-last column
                 DIRECTIONS, MPI_DOUBLE,
                 rank + 1, // Destination rank (right neighbor)
                 321,      // Message tag
                 MPI_COMM_WORLD);
      }
    }

    // Send to the left neighbor
    if (!first) {
      for (int i = 0; i < height; i++) {
        MPI_Send(lbm_mesh_get_cell(mesh, 1, i), // Send from the second column
                 DIRECTIONS, MPI_DOUBLE,
                 rank - 1, // Destination rank (left neighbor)
                 123,      // Message tag
                 MPI_COMM_WORLD);
      }
    }
  }

  // Receive from the left neighbor
  if (!first) {
    for (int i = 0; i < height; i++) {
      MPI_Recv(
          lbm_mesh_get_cell(mesh, 0, i), // Receive into the leftmost column
          DIRECTIONS, MPI_DOUBLE,
          rank - 1, // Source rank (left neighbor)
          321,      // Message tag
          MPI_COMM_WORLD,
          MPI_STATUS_IGNORE // Ignore the status
      );
    }
  }

  // Receive from the right neighbor
  if (!last) {
    for (int i = 0; i < height; i++) {
      MPI_Recv(lbm_mesh_get_cell(mesh, width - 1,
                                 i), // Receive into the rightmost column
               DIRECTIONS, MPI_DOUBLE,
               rank + 1, // Source rank (right neighbor)
               123,      // Message tag
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE // Ignore the status
      );
    }
  }

  // Even ranks receive first, then send
  if (rank % 2 == 0) { // EVEN: Send after receiving
    // Send to the right neighbor
    if (!last) {
      for (int i = 0; i < height; i++) {
        MPI_Send(lbm_mesh_get_cell(mesh, width - 2,
                                   i), // Send from the second-to-last column
                 DIRECTIONS, MPI_DOUBLE,
                 rank + 1, // Destination rank (right neighbor)
                 321,      // Message tag
                 MPI_COMM_WORLD);
      }
    }

    // Send to the left neighbor
    if (!first) {
      for (int i = 0; i < height; i++) {
        MPI_Send(lbm_mesh_get_cell(mesh, 1, i), // Send from the second column
                 DIRECTIONS, MPI_DOUBLE,
                 rank - 1, // Destination rank (left neighbor)
                 123,      // Message tag
                 MPI_COMM_WORLD);
      }
    }
  }
}