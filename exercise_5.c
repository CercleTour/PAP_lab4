/*****************************************************
 AUTHOR : Sébastien Valat
 MAIL : sebastien.valat@univ-grenoble-alpes.fr
 LICENSE : BSD
 YEAR : 2021
 COURSE : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication scheme with
//       8 neighbors using MPI types for non-contiguous
//       side.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
// NEW:
//     - >>> MPI type for non-contiguous cells <<<
//
//////////////////////////////////////////////////////

#include "mpi.h"
#include "src/exercises.h"
#include "src/lbm_struct.h"
#include <math.h>
#include <string.h>

/****************************************************/
// Initialize the communication structure for exercise 5
void lbm_comm_init_ex5(lbm_comm_t *comm, int total_width, int total_height) {
  // Reuse initialization logic from exercise 4
  lbm_comm_init_ex4(comm, total_width, total_height);

  // Create an MPI type for a full row of cells
  MPI_Type_contiguous(comm->width * DIRECTIONS, MPI_DOUBLE, &comm->type);
  MPI_Type_commit(&comm->type);

  // Allocate buffers for top and bottom communication
  comm->buffer_send_up = malloc(comm->width * DIRECTIONS * sizeof(double));
  comm->buffer_recv_up = malloc(comm->width * DIRECTIONS * sizeof(double));
  comm->buffer_send_down = malloc(comm->width * DIRECTIONS * sizeof(double));
  comm->buffer_recv_down = malloc(comm->width * DIRECTIONS * sizeof(double));

  // Check for allocation errors
  if (!comm->buffer_send_up || !comm->buffer_recv_up ||
      !comm->buffer_send_down || !comm->buffer_recv_down) {
    fprintf(stderr, "Error: Failed to allocate memory for buffers.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

/****************************************************/
// Release resources allocated for communication
void lbm_comm_release_ex5(lbm_comm_t *comm) {
  // Reuse release logic from exercise 4
  lbm_comm_release_ex4(comm);

  // Free the custom MPI type
  MPI_Type_free(&comm->type);
}

/****************************************************/
// Directions for 8 neighbors (dx, dy)
static int directions[8][2] = {
    {-1, 0},  // 0 Left
    {1, 0},   // 1 Right
    {0, -1},  // 2 Top
    {0, 1},   // 3 Bottom
    {-1, -1}, // 4 Top-left
    {1, -1},  // 5 Top-right
    {-1, 1},  // 6 Bottom-left
    {1, 1}    // 7 Bottom-right
};

/****************************************************/
// Copy data from the mesh to a buffer
static void copy_to_buffer(double *buffer, int start_x, int start_y, int width,
                           int height, lbm_mesh_t *mesh) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double *cell = lbm_mesh_get_cell(mesh, start_x + x, start_y + y);
      memcpy(buffer + (y * width + x) * DIRECTIONS, cell,
             DIRECTIONS * sizeof(double));
    }
  }
}

/****************************************************/
// Copy data from a buffer to the mesh
static void copy_from_buffer(double *buffer, int start_x, int start_y,
                             int width, int height, lbm_mesh_t *mesh) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double *cell = lbm_mesh_get_cell(mesh, start_x + x, start_y + y);
      memcpy(cell, buffer + (y * width + x) * DIRECTIONS,
             DIRECTIONS * sizeof(double));
    }
  }
}

/****************************************************/
// Get the rank of a neighboring process based on relative position
static inline int get_neighbor_rank(int dx, int dy, lbm_comm_t *comm) {
  int new_x = comm->rank_x + dx;
  int new_y = comm->rank_y + dy;

  // Check if the neighbor is out of bounds
  if (new_x < 0 || new_x >= comm->nb_x || new_y < 0 || new_y >= comm->nb_y) {
    return MPI_PROC_NULL; // No neighbor
  }

  // Calculate the rank of the neighbor
  return new_y * comm->nb_x + new_x;
}

/****************************************************/
// Perform ghost cell exchange using blocking communication
void lbm_comm_ghost_exchange_ex5(lbm_comm_t *comm, lbm_mesh_t *mesh) {
  MPI_Status status;

  // Communication in 8 directions
  for (int i = 0; i < 8; i++) {
    int dx = directions[i][0];
    int dy = directions[i][1];
    int rank = get_neighbor_rank(dx, dy, comm);

    // Skip if there is no neighbor
    if (rank != MPI_PROC_NULL) {
      if (i == 0) { // Left
        MPI_Sendrecv(lbm_mesh_get_cell(mesh, 1, 0), comm->height * DIRECTIONS,
                     MPI_DOUBLE, rank, 0, lbm_mesh_get_cell(mesh, 0, 0),
                     comm->height * DIRECTIONS, MPI_DOUBLE, rank, 0,
                     MPI_COMM_WORLD, &status);
      } else if (i == 1) { // Right
        MPI_Sendrecv(lbm_mesh_get_cell(mesh, comm->width - 2, 0),
                     comm->height * DIRECTIONS, MPI_DOUBLE, rank, 0,
                     lbm_mesh_get_cell(mesh, comm->width - 1, 0),
                     comm->height * DIRECTIONS, MPI_DOUBLE, rank, 0,
                     MPI_COMM_WORLD, &status);
      } else if (i == 2) { // Top
        copy_to_buffer(comm->buffer_send_up, 0, 1, comm->width, 1, mesh);
        MPI_Sendrecv(comm->buffer_send_up, 1, comm->type, rank, 0,
                     comm->buffer_recv_up, 1, comm->type, rank, 0,
                     MPI_COMM_WORLD, &status);
        copy_from_buffer(comm->buffer_recv_up, 0, 0, comm->width, 1, mesh);
      } else if (i == 3) { // Bottom
        copy_to_buffer(comm->buffer_send_down, 0, comm->height - 2, comm->width,
                       1, mesh);
        MPI_Sendrecv(comm->buffer_send_down, 1, comm->type, rank, 0,
                     comm->buffer_recv_down, 1, comm->type, rank, 0,
                     MPI_COMM_WORLD, &status);
        copy_from_buffer(comm->buffer_recv_down, 0, comm->height - 1,
                         comm->width, 1, mesh);
      } else if (i == 4) { // Top-left
        MPI_Sendrecv(lbm_mesh_get_cell(mesh, 1, 1), DIRECTIONS, MPI_DOUBLE,
                     rank, 0, lbm_mesh_get_cell(mesh, 0, 0), DIRECTIONS,
                     MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);
      } else if (i == 5) { // Top-right
        MPI_Sendrecv(lbm_mesh_get_cell(mesh, comm->width - 2, 1), DIRECTIONS,
                     MPI_DOUBLE, rank, 0,
                     lbm_mesh_get_cell(mesh, comm->width - 1, 0), DIRECTIONS,
                     MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);
      } else if (i == 6) { // Bottom-left
        MPI_Sendrecv(lbm_mesh_get_cell(mesh, 1, comm->height - 2), DIRECTIONS,
                     MPI_DOUBLE, rank, 0,
                     lbm_mesh_get_cell(mesh, 0, comm->height - 1), DIRECTIONS,
                     MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);
      } else if (i == 7) { // Bottom-right
        MPI_Sendrecv(lbm_mesh_get_cell(mesh, comm->width - 2, comm->height - 2),
                     DIRECTIONS, MPI_DOUBLE, rank, 0,
                     lbm_mesh_get_cell(mesh, comm->width - 1, comm->height - 1),
                     DIRECTIONS, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);
      }
    }
  }
}