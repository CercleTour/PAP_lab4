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
	lbm_comm_init_ex1(comm, total_width, total_height);
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