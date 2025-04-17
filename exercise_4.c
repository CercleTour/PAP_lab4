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

#include "src/lbm_struct.h"
#include "src/exercises.h"
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void lbm_comm_init_ex4(lbm_comm_t *comm, int total_width, int total_height)
{
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	comm->nb_y = (int)sqrt(size);
	while (size % comm->nb_y != 0)
		comm->nb_y--;

	comm->nb_x = size / comm->nb_y;

	comm->rank_x = rank % comm->nb_x;
	comm->rank_y = rank / comm->nb_x;

	int local_width = total_width / comm->nb_x;
	int local_height = total_height / comm->nb_y;

	comm->width = local_width + 2;
	comm->height = local_height + 2;

	comm->x = comm->rank_x * local_width;
	comm->y = comm->rank_y * local_height;

	int ghost_line_size = (comm->width - 2) * DIRECTIONS;
	comm->buffer_send_up = malloc(ghost_line_size * sizeof(double));
	comm->buffer_recv_up = malloc(ghost_line_size * sizeof(double));
	comm->buffer_send_down = malloc(ghost_line_size * sizeof(double));
	comm->buffer_recv_down = malloc(ghost_line_size * sizeof(double));

	if (!comm->buffer_send_up || !comm->buffer_recv_up ||
		!comm->buffer_send_down || !comm->buffer_recv_down)
	{
		fprintf(stderr, "malloc failed\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	lbm_comm_init_ex1(comm, total_width, total_height);
}

void lbm_comm_release_ex4(lbm_comm_t *comm)
{
	free(comm->buffer_send_up);
	free(comm->buffer_recv_up);
	free(comm->buffer_send_down);
	free(comm->buffer_recv_down);
	comm->buffer_send_up = comm->buffer_recv_up = NULL;
	comm->buffer_send_down = comm->buffer_recv_down = NULL;
}

void getRank(int *rank_x, int *rank_y, int x, int y, int nb_x, int nb_y)
{
	*rank_x = (x + nb_x) % nb_x;
	*rank_y = (y + nb_y) % nb_y;
}

void lbm_comm_ghost_exchange_ex4(lbm_comm_t *comm, lbm_mesh_t *mesh)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int width = comm->width;
	int height = comm->height;
	int rx = comm->rank_x;
	int ry = comm->rank_y;
	int nbx = comm->nb_x;
	int nby = comm->nb_y;

	// Déterminer les voisins (MPI_PROC_NULL si hors grille)
	int left = (rx > 0) ? rank - 1 : MPI_PROC_NULL;
	int right = (rx < nbx - 1) ? rank + 1 : MPI_PROC_NULL;
	int top = (ry > 0) ? rank - nbx : MPI_PROC_NULL;
	int bottom = (ry < nby - 1) ? rank + nbx : MPI_PROC_NULL;

	int top_left = (left != MPI_PROC_NULL && top != MPI_PROC_NULL) ? top - 1 : MPI_PROC_NULL;
	int top_right = (right != MPI_PROC_NULL && top != MPI_PROC_NULL) ? top + 1 : MPI_PROC_NULL;
	int bottom_left = (left != MPI_PROC_NULL && bottom != MPI_PROC_NULL) ? bottom - 1 : MPI_PROC_NULL;
	int bottom_right = (right != MPI_PROC_NULL && bottom != MPI_PROC_NULL) ? bottom + 1 : MPI_PROC_NULL;

	// --- LEFT/RIGHT communication (contiguous) ---
	for (int i = 1; i < height - 1; i++)
	{
		// Send left
		MPI_Sendrecv(
			lbm_mesh_get_cell(mesh, 1, i), DIRECTIONS, MPI_DOUBLE, left, 0,
			lbm_mesh_get_cell(mesh, width - 1, i), DIRECTIONS, MPI_DOUBLE, right, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// Send right
		MPI_Sendrecv(
			lbm_mesh_get_cell(mesh, width - 2, i), DIRECTIONS, MPI_DOUBLE, right, 1,
			lbm_mesh_get_cell(mesh, 0, i), DIRECTIONS, MPI_DOUBLE, left, 1,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// --- TOP/BOTTOM (non-contiguous → buffer) ---
	for (int x = 1; x < width - 1; x++)
	{
		memcpy(&comm->buffer_send_up[(x - 1) * DIRECTIONS], lbm_mesh_get_cell(mesh, x, 1), DIRECTIONS * sizeof(double));
		memcpy(&comm->buffer_send_down[(x - 1) * DIRECTIONS], lbm_mesh_get_cell(mesh, x, height - 2), DIRECTIONS * sizeof(double));
	}

	// Debug: Afficher les buffers avant l'envoi
	if (rank == 0)
	{
		printf("Send_down buffer: ");
		for (int i = 0; i < (width - 2) * DIRECTIONS; ++i)
			printf("%f ", comm->buffer_send_down[i]);
		printf("\n");

		printf("Send_up buffer: ");
		for (int i = 0; i < (width - 2) * DIRECTIONS; ++i)
			printf("%f ", comm->buffer_send_up[i]);
		printf("\n");
	}

	// TOP → BOTTOM
	MPI_Sendrecv(comm->buffer_send_down, (width - 2) * DIRECTIONS, MPI_DOUBLE, bottom, 0,
				 comm->buffer_recv_up, (width - 2) * DIRECTIONS, MPI_DOUBLE, top, 1,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// BOTTOM → TOP
	MPI_Sendrecv(comm->buffer_send_up, (width - 2) * DIRECTIONS, MPI_DOUBLE, top, 2,
				 comm->buffer_recv_down, (width - 2) * DIRECTIONS, MPI_DOUBLE, bottom, 3,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// Debug: Afficher les buffers après la réception
	if (rank == 0)
	{
		printf("Recv_up buffer: ");
		for (int i = 0; i < (width - 2) * DIRECTIONS; ++i)
			printf("%f ", comm->buffer_recv_up[i]);
		printf("\n");

		printf("Recv_down buffer: ");
		for (int i = 0; i < (width - 2) * DIRECTIONS; ++i)
			printf("%f ", comm->buffer_recv_down[i]);
		printf("\n");
	}

	// Copier les données reçues dans le maillage
	for (int x = 1; x < width - 1; x++)
	{
		memcpy(lbm_mesh_get_cell(mesh, x, 0), &comm->buffer_recv_up[(x - 1) * DIRECTIONS], DIRECTIONS * sizeof(double));
		memcpy(lbm_mesh_get_cell(mesh, x, height - 1), &comm->buffer_recv_down[(x - 1) * DIRECTIONS], DIRECTIONS * sizeof(double));
	}

	// --- Corners (diagonales) ---
	// top-left
	MPI_Sendrecv(
		lbm_mesh_get_cell(mesh, 1, 1), DIRECTIONS, MPI_DOUBLE, top_left, 0,
		lbm_mesh_get_cell(mesh, width - 1, height - 1), DIRECTIONS, MPI_DOUBLE, bottom_right, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// top-right
	MPI_Sendrecv(
		lbm_mesh_get_cell(mesh, width - 2, 1), DIRECTIONS, MPI_DOUBLE, top_right, 0,
		lbm_mesh_get_cell(mesh, 0, height - 1), DIRECTIONS, MPI_DOUBLE, bottom_left, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// bottom-left
	MPI_Sendrecv(
		lbm_mesh_get_cell(mesh, 1, height - 2), DIRECTIONS, MPI_DOUBLE, bottom_left, 0,
		lbm_mesh_get_cell(mesh, width - 1, 0), DIRECTIONS, MPI_DOUBLE, top_right, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// bottom-right
	MPI_Sendrecv(
		lbm_mesh_get_cell(mesh, width - 2, height - 2), DIRECTIONS, MPI_DOUBLE, bottom_right, 0,
		lbm_mesh_get_cell(mesh, 0, 0), DIRECTIONS, MPI_DOUBLE, top_left, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
