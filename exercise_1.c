/*****************************************************
AUTHOR  : Sébastien Valat
MAIL    : sebastien.valat@univ-grenoble-alpes.fr
LICENSE : BSD
YEAR    : 2021
COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// GOAL: Implement a 1D communication scheme along
//       X axis with blocking communications.
//
// SUMMARY:
//     - 1D splitting along X
//     - Blocking communications
//
//////////////////////////////////////////////////////

/****************************************************/
#include "mpi.h"
#include "src/exercises.h"
#include "src/lbm_struct.h"
#include <stdbool.h>

/****************************************************/
// Initialize the communication structure for exercise 1
void lbm_comm_init_ex1(lbm_comm_t *comm, int total_width, int total_height) {
  // Get the rank of the current process and the total number of processes
  int rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  // Ensure the total width is divisible by the number of processes
  if (total_width % comm_size != 0) {
    char error_message[256];
    sprintf(error_message,
            "Invalid communicator size! (%d / %d is not an integer)", comm_size,
            total_width);
    fatal(error_message);
  }

  // Set the number of processes along the X and Y dimensions
  comm->nb_x = comm_size;
  comm->nb_y = 1;

  // Set the rank of the current process in the X and Y dimensions
  comm->rank_x = rank;
  comm->rank_y = 0;

  // Calculate the local sub-domain size (including ghost cells)
  comm->width = total_width / comm_size + 2; // +2 for ghost cells
  comm->height = total_height + 2;           // +2 for ghost cells

  // Calculate the absolute position in the global mesh (excluding ghost cells)
  comm->x = (comm->width - 2) * rank; // Starting X position
  comm->y = 0; // Starting Y position (always 0 for 1D splitting)

  // Print the communication structure for debugging
  lbm_comm_print(comm);
}

/****************************************************/
// Perform ghost cell exchange using blocking communication
void lbm_comm_ghost_exchange_ex1(lbm_comm_t *comm, lbm_mesh_t *mesh) {
  // Determine if the current process is the first or last in the X dimension
  bool first = comm->rank_x == 0;
  bool last = comm->rank_x == comm->nb_x - 1;

  // Get the local mesh dimensions
  int height = comm->height;
  int width = comm->width;

  // Send cells to the right neighbor (if not the last process)
  if (!last) {
    for (int i = 0; i < height; i++) {
      MPI_Send(lbm_mesh_get_cell(mesh, width - 2,
                                 i), // Send from the second-to-last column
               DIRECTIONS,           // Number of elements in the buffer
               MPI_DOUBLE,           // Data type
               comm->rank_x + 1,     // Destination rank (right neighbor)
               321,                  // Message tag
               MPI_COMM_WORLD        // Communicator
      );
    }
  }

  // Receive cells from the left neighbor (if not the first process)
  if (!first) {
    for (int i = 0; i < height; i++) {
      MPI_Recv(
          lbm_mesh_get_cell(mesh, 0, i), // Receive into the leftmost column
          DIRECTIONS,       // Maximum number of elements in the buffer
          MPI_DOUBLE,       // Data type
          comm->rank_x - 1, // Source rank (left neighbor)
          321,              // Message tag
          MPI_COMM_WORLD,   // Communicator
          MPI_STATUS_IGNORE // Ignore the status
      );
    }
  }

  // Send cells to the left neighbor (if not the first process)
  if (!first) {
    for (int i = 0; i < height; i++) {
      MPI_Send(lbm_mesh_get_cell(mesh, 1, i), // Send from the second column
               DIRECTIONS,       // Number of elements in the buffer
               MPI_DOUBLE,       // Data type
               comm->rank_x - 1, // Destination rank (left neighbor)
               123,              // Message tag
               MPI_COMM_WORLD    // Communicator
      );
    }
  }

  // Receive cells from the right neighbor (if not the last process)
  if (!last) {
    for (int i = 0; i < height; i++) {
      MPI_Recv(lbm_mesh_get_cell(mesh, width - 1,
                                 i), // Receive into the rightmost column
               DIRECTIONS,           // Maximum number of elements in the buffer
               MPI_DOUBLE,           // Data type
               comm->rank_x + 1,     // Source rank (right neighbor)
               123,                  // Message tag
               MPI_COMM_WORLD,       // Communicator
               MPI_STATUS_IGNORE     // Ignore the status
      );
    }
  }
}
