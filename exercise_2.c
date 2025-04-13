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
#include "src/lbm_struct.h"
#include "src/exercises.h"
#include "mpi.h"
#include <stdbool.h>

/****************************************************/
void lbm_comm_init_ex2(lbm_comm_t * comm, int total_width, int total_height)
{
	//
	// TODO: calculate the splitting parameters for the current task.
	//
	// HINT: You can look in exercise_0.c to get an example for the sequential case.
	//
	int rank;
	int comm_size;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
	
	if (total_width % comm_size != 0) {
		char error_message[256];
		sprintf(error_message, "Invalid communicator size ! (%d / %d is not an integer)", comm_size, total_width);
		fatal(error_message);
	}
	comm->nb_x = comm_size;
	comm->nb_y = 1;
	
	comm->rank_x = rank;
	comm->rank_y = 0;
	
	// TODO : calculate the local sub-domain size (do not forget the 
	//        ghost cells). Use total_width & total_height as starting 
	//        point.
	comm->width = total_width/comm_size + 2;
	comm->height = total_height + 2;
	// + 2 for ghost cells
	
	// TODO : calculate the absolute position in the global mesh.
	//        without accounting the ghost cells
	//        (used to setup the obstable & initial conditions).
	comm->x = (comm->width-2) * rank;
	comm->y = 0;
	
	//if debug print comm
	lbm_comm_print(comm);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex2(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	bool first = comm->rank_x == 0;
	bool last = comm->rank_x == comm->nb_x-1;

	int height = comm->height;
	int width = comm->width;

	int rank = comm->rank_x;

	if (rank % 2 != 0) { // ODE = Send befor
		// Send to right
		if (!last) {
			for (int i = 0; i < height; i++) {
				MPI_Send(
					lbm_mesh_get_cell(mesh, width - 2, i),
					DIRECTIONS,
					MPI_DOUBLE,
					rank + 1,
					321,
					MPI_COMM_WORLD
				);
			}
		}
		
		// Send to left
		if (!first) {
			for (int i = 0; i < height; i++) {
				MPI_Send(
					lbm_mesh_get_cell(mesh, 1, i),
					DIRECTIONS,
					MPI_DOUBLE,
					rank - 1,
					123,
					MPI_COMM_WORLD
				);
			}
		}
	}
	
	// Receive from left
	if (!first) {
		for (int i = 0; i < height; i++) {
			MPI_Recv(
				lbm_mesh_get_cell(mesh, 0, i),
				DIRECTIONS,
				MPI_DOUBLE,
				rank - 1,
				321,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE
			);
		}
	}
	
	// Receive from right
	if (!last) {
		for (int i = 0; i < height; i++) {
			MPI_Recv(
				lbm_mesh_get_cell(mesh, width - 1, i),
				DIRECTIONS,
				MPI_DOUBLE,
				rank + 1,
				123,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE
			);
		}
	}
	if (rank % 2 == 0) { // EVEN : send after
		// Send to right
		if (!last) {
			for (int i = 0; i < height; i++) {
				MPI_Send(
					lbm_mesh_get_cell(mesh, width - 2, i),
					DIRECTIONS,
					MPI_DOUBLE,
					rank + 1,
					321,
					MPI_COMM_WORLD
				);
			}
		}
		
		// Send to left
		if (!first) {
			for (int i = 0; i < height; i++) {
				MPI_Send(
					lbm_mesh_get_cell(mesh, 1, i),
					DIRECTIONS,
					MPI_DOUBLE,
					rank - 1,
					123,
					MPI_COMM_WORLD
				);
			}
		}
	}
}