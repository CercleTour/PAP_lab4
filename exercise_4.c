/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication scheme with
//       8 neighbors using manual copy for non
//       contiguous side and blocking communications
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
//     - Manual copy for non continguous cells
//
//////////////////////////////////////////////////////

/****************************************************/

#include "mpi.h"
#include "src/lbm_struct.h"
#include "src/exercises.h" 


//////////////////////////////////////////////////////
#include <math.h>
#include <string.h>

/****************************************************/
void lbm_comm_init_ex4(lbm_comm_t * comm, int total_width, int total_height)
{
    //
    // TODO: calculate the splitting parameters for the current task.
    //
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // TODO: calculate the number of tasks along X axis and Y axis.

    comm->nb_x = (int)(sqrt(size));
    comm->nb_y = (int)(size / comm->nb_x);

    if (comm->nb_x * comm->nb_y != size || total_width % comm->nb_x != 0 || total_height % comm->nb_y != 0) {
        fprintf(stderr, "Error: Invalid number of processes or domain size.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // TODO: calculate the current task position in the splitting
    comm->rank_x = rank % comm->nb_x;
    comm->rank_y = rank / comm->nb_x;

    // TODO : calculate the local sub-domain size (do not forget the 
    //        ghost cells). Use total_width & total_height as starting 
    //        point.
    comm->width = total_width / comm->nb_x + 2;
    comm->height = total_height / comm->nb_y + 2;

    // TODO : calculate the absolute position  (in cell number) in the global mesh.
    //        without accounting the ghost cells
    //        (used to setup the obstable & initial conditions).
    comm->x = (comm->rank_x * total_width) / comm->nb_x;
    comm->y = (comm->rank_y * total_height) / comm->nb_y;

    //OPTIONAL : if you want to avoid allocating temporary copy buffer
    //           for every step :
    //comm->buffer_recv_down, comm->buffer_recv_up, comm->buffer_send_down, comm->buffer_send_up
    comm->buffer_recv_down = malloc(comm->width * DIRECTIONS * sizeof(double));
    comm->buffer_recv_up = malloc(comm->width * DIRECTIONS * sizeof(double));
    comm->buffer_send_down = malloc(comm->width * DIRECTIONS * sizeof(double));
    comm->buffer_send_up = malloc(comm->width * DIRECTIONS * sizeof(double));
    if (!comm->buffer_recv_down || !comm->buffer_recv_up || !comm->buffer_send_down || !comm->buffer_send_up) {
        fprintf(stderr, "Error: Failed to allocate memory for buffers.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //if debug print comm
    //lbm_comm_print(comm);
}

/****************************************************/
void lbm_comm_release_ex4(lbm_comm_t * comm)
{
    //free allocated ressources
    free(comm->buffer_recv_down);
    free(comm->buffer_recv_up);
    free(comm->buffer_send_down);
    free(comm->buffer_send_up);
}

void copy_to_buffer(double *buffer, int start_x, int start_y, int width, int height, lbm_mesh_t *mesh) {

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double *cell = lbm_mesh_get_cell(mesh, start_x + x, start_y + y);
            memcpy(buffer + (y * width + x) * DIRECTIONS, cell, DIRECTIONS * sizeof(double));
        }
    }
}

void copy_from_buffer(double *buffer, int start_x, int start_y, int width, int height, lbm_mesh_t *mesh) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double *cell = lbm_mesh_get_cell(mesh, start_x + x, start_y + y);
            memcpy(cell, buffer + (y * width + x) * DIRECTIONS, DIRECTIONS * sizeof(double));
        }
    }
}

int get_neighbor_rank(int dx, int dy, lbm_comm_t *comm) {
    //printf("get_neighbor_rank : rank %d dx %d dy %d\n", comm->rank_x, dx, dy);
    int new_x = comm->rank_x + dx;
    int new_y = comm->rank_y + dy;
    if (new_x < 0 || new_x >= comm->nb_x || new_y < 0 || new_y >= comm->nb_y) {
        return MPI_PROC_NULL; // No neighbor
    }
    //printf("get_neighbor_rank : rank %d new_x %d new_y %d\n", comm->rank_x, new_x, new_y);
    return new_y * comm->nb_x + new_x;
}


/****************************************************/
void lbm_comm_ghost_exchange_ex4(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    int ranks[8];
    
    // Left/Right communications
    ranks[0] = get_neighbor_rank(-1, 0, comm); // Left
    ranks[1] = get_neighbor_rank(1, 0, comm);  // Right

    // Left-right exchange
    // When ranks[0] is MPI_PROC_NULL (leftmost process), no actual send/receive will happen
    MPI_Sendrecv(
        lbm_mesh_get_cell(mesh, 1, 0), comm->height * DIRECTIONS, MPI_DOUBLE, ranks[0], 0,
        lbm_mesh_get_cell(mesh, 0, 0), comm->height * DIRECTIONS, MPI_DOUBLE, ranks[0], 0,
        MPI_COMM_WORLD, &status
    );
    
    // When ranks[1] is MPI_PROC_NULL (rightmost process), no actual send/receive will happen
    MPI_Sendrecv(
        lbm_mesh_get_cell(mesh, comm->width - 2, 0), comm->height * DIRECTIONS, MPI_DOUBLE, ranks[1], 0,
        lbm_mesh_get_cell(mesh, comm->width - 1, 0), comm->height * DIRECTIONS, MPI_DOUBLE, ranks[1], 0,
        MPI_COMM_WORLD, &status
    );

    // Top/Bottom communications
    ranks[2] = get_neighbor_rank(0, -1, comm); // Top
    ranks[3] = get_neighbor_rank(0, 1, comm);  // Bottom

    // Top exchange
    copy_to_buffer(comm->buffer_send_up, 0, 1, comm->width, 1, mesh);
    MPI_Sendrecv(
        comm->buffer_send_up, comm->width * DIRECTIONS, MPI_DOUBLE, ranks[2], 0,
        comm->buffer_recv_up, comm->width * DIRECTIONS, MPI_DOUBLE, ranks[2], 0,
        MPI_COMM_WORLD, &status
    );
    // Only copy from buffer if we have a top neighbor
    if (ranks[2] != MPI_PROC_NULL) {
        copy_from_buffer(comm->buffer_recv_up, 0, 0, comm->width, 1, mesh);
    }

    // Bottom exchange
    copy_to_buffer(comm->buffer_send_down, 0, comm->height - 2, comm->width, 1, mesh);
    MPI_Sendrecv(
        comm->buffer_send_down, comm->width * DIRECTIONS, MPI_DOUBLE, ranks[3], 0,
        comm->buffer_recv_down, comm->width * DIRECTIONS, MPI_DOUBLE, ranks[3], 0,
        MPI_COMM_WORLD, &status
    );
    // Only copy from buffer if we have a bottom neighbor
    if (ranks[3] != MPI_PROC_NULL) {
        copy_from_buffer(comm->buffer_recv_down, 0, comm->height - 1, comm->width, 1, mesh);
    }

    // Diagonal communications
    ranks[4] = get_neighbor_rank(-1, -1, comm); // Top-left
    ranks[5] = get_neighbor_rank(1, -1, comm);  // Top-right
    ranks[6] = get_neighbor_rank(-1, 1, comm);  // Bottom-left
    ranks[7] = get_neighbor_rank(1, 1, comm);   // Bottom-right

    // Top-left exchange
    MPI_Sendrecv(
        lbm_mesh_get_cell(mesh, 1, 1), DIRECTIONS, MPI_DOUBLE, ranks[4], 0,
        lbm_mesh_get_cell(mesh, 0, 0), DIRECTIONS, MPI_DOUBLE, ranks[4], 0,
        MPI_COMM_WORLD, &status
    );

    // Top-right exchange
    MPI_Sendrecv(
        lbm_mesh_get_cell(mesh, comm->width - 2, 1), DIRECTIONS, MPI_DOUBLE, ranks[5], 0,
        lbm_mesh_get_cell(mesh, comm->width - 1, 0), DIRECTIONS, MPI_DOUBLE, ranks[5], 0,
        MPI_COMM_WORLD, &status
    );

    // Bottom-left exchange
    MPI_Sendrecv(
        lbm_mesh_get_cell(mesh, 1, comm->height - 2), DIRECTIONS, MPI_DOUBLE, ranks[6], 0,
        lbm_mesh_get_cell(mesh, 0, comm->height - 1), DIRECTIONS, MPI_DOUBLE, ranks[6], 0,
        MPI_COMM_WORLD, &status
    );

    // Bottom-right exchange
    MPI_Sendrecv(
        lbm_mesh_get_cell(mesh, comm->width - 2, comm->height - 2), DIRECTIONS, MPI_DOUBLE, ranks[7], 0,
        lbm_mesh_get_cell(mesh, comm->width - 1, comm->height - 1), DIRECTIONS, MPI_DOUBLE, ranks[7], 0,
        MPI_COMM_WORLD, &status
    );
}