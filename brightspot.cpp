// g++ -g -Wall -std=c++11 brightspot.cpp -o a.out -fopenmp -lpthread

#include <iostream>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <vector>

using namespace std;

void printMatrix(float* mat, int limit, int col){
        for(int i = 0; i < limit; i++){
                for(int j = 0; j < limit; j++){
                        cout << mat[i * col + j] << " ";
                        if(j == limit - 1){
                        cout << endl;
                        }
                }
        }
}


int brights(float* pix, int n, int k, float thresh){
	float* discrete = new float[n * n];

	#pragma omp parallel
	{
		int me = omp_get_thread_num();
		int all = omp_get_num_threads();

		int lower = me*n/all;
		int upper = (me+1)*n/all;
		
		for(int i = lower; i < upper; i++){
                	float lastElement = 0;
                	for(int j = 0; j < n; j++){
                        	if(pix[i*n+j] >= thresh){
                                	lastElement += 1;
                                	discrete[i*n+j] = lastElement;
                        	} else {
                                	lastElement = 0;
                                	discrete[i*n+j] = 0;
                        	}
                	}
        	}
	}

	int count = 0;
	#pragma omp parallel
	{
		int me = omp_get_thread_num();
                int all = omp_get_num_threads();

		int nn = n-k+1;
		int lower = me*nn/all;
                int upper = (me+1)*nn/all;

		int localCount = 0;
		for(int i = lower; i < upper; i++){
                        for(int j = k-1; j < n; j++){
				if(discrete[i*n+j] >= k){
					bool anyDark = false;
					for(int K = 1; K < k; K++){
						if(discrete[(i+K)*n+j] < k){
							anyDark = true;
							break;
						}
					}
					if(anyDark==false){
						localCount += 1;
					}
				} else {
					j += k - discrete[i*n+j] - 1;
				}
                        }
                }
		#pragma omp critical
		{
			count += localCount;
		}
	
	}

	cout << "===== discrete =====" << endl;
        printMatrix(discrete, 10, n);
	
	cout << "===== n =====" << endl;
	cout << n << endl;

	cout << "===== k =====" << endl;
        cout << k << endl;

	cout << "===== bright block count =====" << endl;
	cout << count << endl;
	return 0;
}

int main(){
	int n = 10;
	int k = 3;
	float thresh = 0.3;
	float* pix = new float[n * n];
	int seed = time(NULL);
    	srand(seed);
	for(int i = 0; i < n * n; i++){
		pix[i] = (float)rand() / (float)RAND_MAX;
	}
	cout << "===== pix =====" << endl;
	printMatrix(pix, 10, n);
	auto begin = std::chrono::high_resolution_clock::now();
	brights(pix, n, k, thresh);
	auto end = std::chrono::high_resolution_clock::now();
	cout << "===== time =====" << endl;
	std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / float(1000000) << "ms" << std::endl;
}
