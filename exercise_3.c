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
#include "src/lbm_struct.h"
#include "src/exercises.h"
#include "mpi.h"
#include <stdbool.h>

/****************************************************/
void lbm_comm_init_ex3(lbm_comm_t * comm, int total_width, int total_height)
{
	//we use the same implementation than ex1
	lbm_comm_init_ex1(comm, total_width, total_height);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex3(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	bool first = comm->rank_x == 0;
	bool last = comm->rank_x == comm->nb_x-1;

	int height = comm->height;
	int width = comm->width;
	
	int rank = comm->rank_x;

	// Pour attendre la fin de toutes les commuications 
	// 4 * hauteur car :
	// - il faut attendre la fin de toutes les com's envoyé à gauche et à droite (pas couper avant d'avoir tout envoyé)
	//    - 2 * hauteur
	// - il faut attendre la fin de toutes les réceptions à gauche et à droite (pas couper avant d'avoir tout recu)
	//    - 2 * hauteur
	MPI_Request requests[4 * height];
	int req_idx = 0;

	// Mise en place des receptions à gauche
	if (!first) {
		for (int i = 0; i < height; i++) {
			MPI_Irecv(
				lbm_mesh_get_cell(mesh, 0, i),
				DIRECTIONS,
				MPI_DOUBLE,
				rank - 1,
				321,
				MPI_COMM_WORLD,
				&requests[req_idx++]
			);
		}
	}
	// Mise en place des receptions à droite
	if (!last) {
		for (int i = 0; i < height; i++) {
			MPI_Irecv(
				lbm_mesh_get_cell(mesh, width - 1, i),
				DIRECTIONS,
				MPI_DOUBLE,
				rank + 1,
				123,
				MPI_COMM_WORLD,
				&requests[req_idx++]
			);
		}
	}

	// Send à droite
	if (!last) {
		for (int i = 0; i < height; i++) {
			MPI_Isend(
				lbm_mesh_get_cell(mesh, width - 2, i),
				DIRECTIONS,
				MPI_DOUBLE,
				rank + 1,
				321,
				MPI_COMM_WORLD,
				&requests[req_idx++]
			);
		}
	}
	// Send à gauche
	if (!first) {
		for (int i = 0; i < height; i++) {
			MPI_Isend(
				lbm_mesh_get_cell(mesh, 1, i),
				DIRECTIONS,
				MPI_DOUBLE,
				rank - 1,
				123,
				MPI_COMM_WORLD,
				&requests[req_idx++]
			);
		}
	}

	// Attendre la fin de toutes les communications : send et recv
	MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);
}