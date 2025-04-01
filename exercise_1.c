/*****************************************************
AUTHOR  : Sébastien Valat
MAIL    : sebastien.valat@univ-grenoble-alpes.fr
LICENSE : BSD
YEAR    : 2021
COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
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
#include "src/lbm_struct.h"
#include "src/exercises.h"
#include "mpi.h"
#include <stdbool.h>

/****************************************************/
void lbm_comm_init_ex1(lbm_comm_t * comm, int total_width, int total_height)
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
void lbm_comm_ghost_exchange_ex1(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	//
	// TODO: Implement the 1D communication with blocking MPI functions (MPI_Send & MPI_Recv)
	//
	// To be used:
	//    - DIRECTIONS: the number of doubles composing a cell
	//    - double[DIRECTIONS] lbm_mesh_get_cell(mesh, x, y): function to get the address of a particular cell.
	//    - comm->width : The with of the local sub-domain (containing the ghost cells)
	//    - comm->height : The height of the local sub-domain (containing the ghost cells)
	
	bool first = comm->rank_x == 0;
	bool last = comm->rank_x == comm->nb_x-1;

	int height = comm->height;
	int width = comm->width;
	
	// Envoyer ses cells vers la droite : 
	if(!last) {
		for(int i = 0; i < height; i++){
			MPI_Send(
				lbm_mesh_get_cell(mesh, width-2, i), // pointer to buffer
				DIRECTIONS, // number of elements of particular type
				MPI_DOUBLE, // type of elements in buffer
				comm->rank_x+1, // destination rank
				321, // tag for send (must match tag for receive)
				MPI_COMM_WORLD // communicator to use
			);
		}
	}
	
	// Recevoir de la gauche
	if(!first) {
		// TODO : Utiliser lbm_mesh_get_cell dans le recv dans 
		for(int i = 0; i < height; i++) {
			MPI_Recv(
				lbm_mesh_get_cell(mesh, 0 , i), // pointer to buffer to write data to
				DIRECTIONS, // max. capacity of elements in buffer
				MPI_DOUBLE, // type of elements in buffer
				comm->rank_x-1, // source rank
				321, // tag for send (must match tag for receive)
				MPI_COMM_WORLD, // communicator to use,
				MPI_STATUS_IGNORE // don’t return statys
			);
		}
	}
	
	// Envoyer ses cellules à gauche donc pas pour le 1er
	if(!first) {
		for(int i = 0; i < height; i++){
			MPI_Send(
				lbm_mesh_get_cell(mesh, 1, i), // pointer to buffer
				DIRECTIONS, // number of elements of particular type
				MPI_DOUBLE, // type of elements in buffer
				comm->rank_x-1, // destination rank
				123, // tag for send (must match tag for receive)
				MPI_COMM_WORLD // communicator to use
			);
		}
	}
	
	// Recevoir de la droite
	if(!last) {
		for(int i = 0; i < height; i++) {
			MPI_Recv(
				lbm_mesh_get_cell(mesh, width-1, i), // pointer to buffer to write data to
				DIRECTIONS, // max. capacity of elements in buffer
				MPI_DOUBLE, // type of elements in buffer
				comm->rank_x+1, // source rank
				123, // tag for send (must match tag for receive)
				MPI_COMM_WORLD, // communicator to use,
				MPI_STATUS_IGNORE // don’t return statys
			);
		}
	}
	
	
	//example to access cell
	//double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
	//double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);
}
