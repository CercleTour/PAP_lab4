/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication with non-blocking
//       messages.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - MPI type for non-contiguous cells
// NEW:
//     - Non-blocking communications
//
//////////////////////////////////////////////////////

#include "mpi.h"
#include "src/exercises.h"
#include "src/lbm_struct.h"
#include <string.h>

/****************************************************/
// Initialize the communication structure for exercise 6
void lbm_comm_init_ex6(lbm_comm_t *comm, int total_width, int total_height) {
  // Reuse the initialization logic from exercise 5
  lbm_comm_init_ex5(comm, total_width, total_height);
}

/****************************************************/
// Release resources allocated for communication
void lbm_comm_release_ex6(lbm_comm_t *comm) {
  // Reuse the release logic from exercise 5
  lbm_comm_release_ex5(comm);
}

// Directions array for neighbor ranks
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
// Perform ghost cell exchange using non-blocking communication
void lbm_comm_ghost_exchange_ex6(lbm_comm_t *comm, lbm_mesh_t *mesh) {
    MPI_Request requests[16]; // 8 send + 8 recv
    int req_id = 0;

    // Get ranks for each direction
    int neighbors[8];
    for (int i = 0; i < 8; ++i) {
        neighbors[i] = get_neighbor_rank(directions[i][0], directions[i][1], comm);
    }

    // Left/Right/Top/Bottom communication

    // Left
    if (neighbors[0] != MPI_PROC_NULL) {
        for (int y = 0; y < comm->height; y++) {
            MPI_Irecv(lbm_mesh_get_cell(mesh, 0, y), DIRECTIONS, MPI_DOUBLE,
                      neighbors[0], 100, MPI_COMM_WORLD, &requests[req_id++]);
            MPI_Isend(lbm_mesh_get_cell(mesh, 1, y), DIRECTIONS, MPI_DOUBLE,
                      neighbors[0], 101, MPI_COMM_WORLD, &requests[req_id++]);
        }
    }

    // Right
    if (neighbors[1] != MPI_PROC_NULL) {
        for (int y = 0; y < comm->height; y++) {
            MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1, y), DIRECTIONS, MPI_DOUBLE,
                      neighbors[1], 101, MPI_COMM_WORLD, &requests[req_id++]);
            MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2, y), DIRECTIONS, MPI_DOUBLE,
                      neighbors[1], 100, MPI_COMM_WORLD, &requests[req_id++]);
        }
    }

    // Top
    if (neighbors[2] != MPI_PROC_NULL) {
        copy_to_buffer(comm->buffer_send_up, 0, 1, comm->width, 1, mesh);
        MPI_Irecv(comm->buffer_recv_up, 1, comm->type, neighbors[2], 200, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(comm->buffer_send_up, 1, comm->type, neighbors[2], 201, MPI_COMM_WORLD, &requests[req_id++]);
    }

    // Bottom
    if (neighbors[3] != MPI_PROC_NULL) {
        copy_to_buffer(comm->buffer_send_down, 0, comm->height - 2, comm->width, 1, mesh);
        MPI_Irecv(comm->buffer_recv_down, 1, comm->type, neighbors[3], 201, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(comm->buffer_send_down, 1, comm->type, neighbors[3], 200, MPI_COMM_WORLD, &requests[req_id++]);
    }

    // Wait for cardinal communication to complete
    MPI_Waitall(req_id, requests, MPI_STATUSES_IGNORE);
    req_id = 0;

    // Copy received data from buffers to the mesh
    if (neighbors[2] != MPI_PROC_NULL) {
        copy_from_buffer(comm->buffer_recv_up, 0, 0, comm->width, 1, mesh);
    }
    if (neighbors[3] != MPI_PROC_NULL) {
        copy_from_buffer(comm->buffer_recv_down, 0, comm->height - 1, comm->width, 1, mesh);
    }

    // Diagonal communication
    // Top-Left
    if (neighbors[4] != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), DIRECTIONS, MPI_DOUBLE,
                  neighbors[4], 300, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(lbm_mesh_get_cell(mesh, 1, 1), DIRECTIONS, MPI_DOUBLE,
                  neighbors[4], 301, MPI_COMM_WORLD, &requests[req_id++]);
    }

    // Top-Right
    if (neighbors[5] != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1, 0), DIRECTIONS, MPI_DOUBLE,
                  neighbors[5], 301, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2, 1), DIRECTIONS, MPI_DOUBLE,
                  neighbors[5], 300, MPI_COMM_WORLD, &requests[req_id++]);
    }

    // Bottom-Left
    if (neighbors[6] != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh, 0, comm->height - 1), DIRECTIONS, MPI_DOUBLE,
                  neighbors[6], 400, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(lbm_mesh_get_cell(mesh, 1, comm->height - 2), DIRECTIONS, MPI_DOUBLE,
                  neighbors[6], 401, MPI_COMM_WORLD, &requests[req_id++]);
    }

    // Bottom-Right
    if (neighbors[7] != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1, comm->height - 1), DIRECTIONS, MPI_DOUBLE,
                  neighbors[7], 401, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2, comm->height - 2), DIRECTIONS, MPI_DOUBLE,
                  neighbors[7], 400, MPI_COMM_WORLD, &requests[req_id++]);
    }

    // Wait for diagonal communication to complete
    MPI_Waitall(req_id, requests, MPI_STATUSES_IGNORE);
}
