#include <iostream>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

void printMatrix(int *mat, int nc, int rowLimit, int colLimit){
	for(int i = 0; i < rowLimit; i++){
		for(int j = 0; j < colLimit; j++){
			cout << mat[i*nc+j] << " ";
		}	
		cout << endl;
	}
}

void findpaths(int *adjm, int n, int k, int *paths, int *numpaths){
	// create a structure to store powers of adj matrix
	int n2 = n * n;
	int *adjPowers = new int[k * n2];
	memcpy(adjPowers, adjm, sizeof(int)*n2);
	*numpaths = 0;

	// calculate powers of adj matrix and save each power into *adjPowers
	for(int kk = 1; kk < k; kk++){
		#pragma omp parallel for
		for(int i = 0; i < n; i++){
                	for(int j = 0; j < n; j++){
                        	int sum = 0;
                        		for(int z = 0; z < n; z++)
                                		sum = sum + adjPowers[(kk-1)*n2+i*n+z] * adjm[z*n+j];
                        	adjPowers[kk*n2+i*n+j] = sum;
                	}	
        	}
	}

	// calculate number of paths
	for(int i = (k-1)*n2; i < (k-1)*n2+n2; i++){
		*numpaths = *numpaths +  adjPowers[i];
	}	
	
	// fill in the starting and ending points of paths in *paths
	int pathRow = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int localNumPaths = adjPowers[(k-1)*n2 + i*n + j];
			for(int z = 0; z < localNumPaths; z++){
				paths[pathRow*(k+1)] = i;
				paths[pathRow*(k+1)+k] = j;
				pathRow++;	
			}
		}
	}

	// cannot parallel easily from here
	// need to calculate cumsum to divide the work
	int *cumsum = new int[n2];
	int tmp = 0;
	for(int i = 0; i < n2; i++){
		cumsum[i] = tmp +  adjPowers[(k-1)*n2+i];
		tmp = cumsum[i];
	}
	
	vector< int > lower;
	vector< int > upper;
	int MAX_TH = 8;
	if(n < 100){
		 MAX_TH = 4;
	}
	
	for(int i = 0; i < MAX_TH; i++){
		lower.push_back(cumsum[(n/MAX_TH)*i*n]);
		upper.push_back(cumsum[(n/MAX_TH)*(i+1)*n]);
	}
	lower[0] = 0;
	upper[MAX_TH-1] = *numpaths;

	// fill the the middle paths
	for(int pathCol = 1; pathCol < k; pathCol++){
		#pragma omp parallel num_threads(MAX_TH)
		{
		int me = omp_get_thread_num();
		for(int pathRow = lower[me]; pathRow < upper[me]; pathRow++){
			int filled = 0;
			int outerStart = paths[pathRow*(k+1) + pathCol - 1];
			int outerEnd = paths[pathRow*(k+1) + k];
			
			// find all candidates that outerStart can go to
			// for example, the outer path is from 0 to 1 of length 3
			// here, we find all candidates 0 can go to
			vector< int > candidates;
			for(int z = 0; z < n; z++){
				if(adjm[outerStart*n+z] >= 1){
					candidates.push_back(z);
				}
			}

			// from those candidates that 0 can go to,
			// see which ones can go to 1 with length 3-1 = 2
			vector< int > goodCandidates;
			for(int z = 0; z < (signed)candidates.size(); z++){
				int new_k = k - pathCol - 1;
				int possible_path = adjPowers[new_k*n2 +  candidates[z]*n + outerEnd];
				if(possible_path >= 1){
					for(int zz = 0; zz < possible_path; zz++){
						paths[(pathRow+filled)*(k+1)+pathCol] = candidates[z];
						filled++;
						goodCandidates.push_back(candidates[z]);
					}
				} 
			}	
			pathRow = pathRow + filled - 1;
		}
		}
	}
}

int main(){
	int n = 100;
	int k = 3;
	int *adjm = new int[n * n];
	int *paths = new int[12766290 * (k+1)];
	int np = 99;
	int *numpaths = &np;
	
	srand(1);
  	for (int i = 0; i < n*n; i++)
    		adjm[i] = rand()%2;	
	auto begin = std::chrono::high_resolution_clock::now();
	findpaths(adjm, n, k, paths, numpaths);	
	auto end = std::chrono::high_resolution_clock::now();

	cout << "===== time =====" << endl;
  	cout << chrono::duration_cast<chrono::nanoseconds>(end-begin).count() / float(1000000) << "ms" << endl;	
	cout << "===== num paths =====" << endl;
        cout << *numpaths << endl;

        cout << "===== paths =====" << endl;
        printMatrix(paths, k+1, 10, k+1);

}	
