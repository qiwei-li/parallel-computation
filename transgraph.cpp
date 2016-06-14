#include <mpi.h>
#include <vector>
#include<iostream>

using namespace std;

int nnodes;
int me;

// need to have the following lines in main !!!!
//
// MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
// MPI_Comm_rank(MPI_COMM_WORLD, &me);

int *transgraph(int *adjm, int n, int *nout){
	// create a structure which holds value and location information from adjm
	int* full = new int[n*n*3];
	// create a structure which holds results
	int* res = new int[n*n*2];

	if(me==0){
		*nout = 0;
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				full[(i*n+j)*3+0] = adjm[i*n+j];
				full[(i*n+j)*3+1] = i;
				full[(i*n+j)*3+2] = j;
			}
		}

		int lenchunk;	
		vector<int> startv;
                vector<int> endv;

		MPI_Status status;
		lenchunk = n*n / (nnodes-1);
		
		// fill in start and end points		
		for(int i = 0; i < nnodes - 1; i++){
			startv.push_back(i*lenchunk);
		} 
		for(int i = 0; i < nnodes - 2; i++){
			endv.push_back((i+1)*lenchunk);
		}
		endv.push_back(n*n);
		
		// send down to pipe
		for(int i = 1; i < nnodes; i++){
			MPI_Send(&full[startv[i-1]*3], (endv[i-1]-startv[i-1])*3, MPI_INT, i, 0, MPI_COMM_WORLD); // 0 pipe
		}
		
		int k = 0;
		// receive from workers
		for(int i = 1; i < nnodes; i++){
			MPI_Recv(res+k, 10000, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_INT, &lenchunk);
			k += lenchunk;
		}

		*nout = k/2;

	} else {
		int lenchunk;
		MPI_Status status;
		MPI_Recv(full, 10000, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &lenchunk);
		
		int count = 0;
		// check which one is non zero and store location in res
		for(int i = 0; i < lenchunk/3; i++){
			if(full[i*3] != 0){
				res[count++] = full[i*3+1];
				res[count++] = full[i*3+2];
			}
		}
		
		MPI_Send(res, count, MPI_INT, 0, 1, MPI_COMM_WORLD); // 1 return to manager
	}

	return res;
}

