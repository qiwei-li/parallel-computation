#include <stdio.h>
#include <iostream>
#include <time.h>

// this requires c++ 11
// #include <chrono>

using namespace std;

//
// compile command on CSIF: /usr/local/cuda-5.5/bin/nvcc gpuquad.cu
//

// one col of A per thread
// both multiplications are done in this kernel
// no shared memory, the cost of setting up shared memory exceeds the speedup
__global__
void quadMultKernel(float *u, float *a, float *res, int n)
{
	// compute my col of a
  int bID = blockDim.x * blockIdx.x + threadIdx.x;

  // this if clause is necessary!
  // I opted for adding one extra block with idle threads when n is not a power of 2
  // the if ensures we only compute n rows
  if (bID < n)
  {
  	// for all of vector u
	  float sum = 0.0;
	  // get dot product with my col of a
	  for (int i = 0; i < n; i++) 
	  {
	    sum += u[i] * a[bID * n + i];
	  }
	  // fill in res
	  res[bID] = sum * u[bID];
	}
}


//
// This function is called for all matices with 256 <= n <= number of threads supported by CUDA card
// for smaller matrices the CPU version is called
//
float gpuquadnormal(float *a, int n, float *u) 
{
  float *d_res, *res;
  float *d_a, *d_u;
  float quad = 0;

  res = new float[n];

  // this number is hardcoded
  // based on fastest performance for various test runs, 256 was giving the best results
  unsigned int threadsPerBlockX = 256;
  // compute grid size based on number of threads for each block
  // silent cast to uint happnening here
  // this could be avoided using ceil, but that requires a math include, potentially reducing performance
  unsigned int gridSizeX = n / threadsPerBlockX;
  // I just compute the ceil manually here:
  if (n % threadsPerBlockX != 0)
  {
  	gridSizeX++;
  }

  // set up the dimensions
  dim3 dimGridN(gridSizeX, 1);
  dim3 dimBlock(threadsPerBlockX, 1, 1);

  cudaMalloc((void**)&d_a, n * n * sizeof(float));
  cudaMemcpy(d_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_u, n * sizeof(float));
  cudaMemcpy(d_u, u, n * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_res, n * sizeof(float));

  quadMultKernel<<<dimGridN, dimBlock>>>(d_u, d_a, d_res, n);

  cudaMemcpy(res, d_res, n * sizeof(float), cudaMemcpyDeviceToHost);

  // sequential CPU add -> faster then all kernels for small array sizes (< 50.000 elements)
  // no point in adding an advanced kernel
  for (int i = 0; i < n; i++)
  {
  	quad += res[i];
  }

  cudaFree(d_res);
  cudaFree(d_a);
  cudaFree(d_u);
  
  return quad;
}

// CPU version of matrix multiplication
// Faster then the kernels for very small matrices (n < 256)
void matrixMult(float* pOne, float* pTwo, int pOneR, int pOneC, int pTwoR, int pTwoC, float* pRes, int pResR, int pResC)
{
  // for each element in the resulting matrix
  for (int i = 0; i < pOneR; i++)
  {
    for (int j = 0; j < pResC; j++)
    {
      float sum = 0;

      for (int k = 0; k < pOneC; k++)
      {
        sum += pOne[i * pOneC + k] * pTwo[ k * pTwoC + j];
      }

      // fill in the resulting matrix
      pRes[i * pResC + j] = sum;
    }
  }
}

//
// If the matrix is small, we achieve better performance by simply computing the quad on the CPU
// Our GPU implementation supports matrixes as small as 1x1, but setting up a kernel for a 1x1 matrix is useless
// The cutoff for using the GPU is a matrix with at least 256 rows and columns 
//
float gpuquad(float *a, int n, float *u)
{
	if (n < 256)
	{
		float *quad = new float;
		float *res = new float[n];
		matrixMult(u, a, 1, n, n, n, res, 1, n);
  	matrixMult(res, u, 1, n, n, 1, quad, 1, 1);
  	return quad[0];
	}
	return gpuquadnormal(a, n, u);
	
}
/*
int main(int argc, char** argv) 
{
  int n = 128 * 128 + 1;
  float *quad = new float;
  float *a = new float[n*n]; // symmetric matrix
  float *u = new float[n];   // vector
  float *res = new float[n];
  srand(3); // random seed

  for (int i = 0; i < n; i++) 
  {
    u[i] = (float)(rand());
  }

  for (int i = 0; i < n; i++)
  {
    for (int j = i; j < n; j++)
    {
      float val = 1; //(float)(rand() % 10);
      a[i*n + j] = val;
      a[j*n + i] = val;
    } // build symmetric matrix
  }


  cout << "Our quad: " << gpuquad(a, n, u) << endl;
  matrixMult(u, a, 1, n, n, n, res, 1, n);
  matrixMult(res, u, 1, n, n, 1, quad, 1, 1);
  cout << "Correct quad: " << quad[0] << endl;
*/
	// THE REQUIRES C++11, uncomment for time estimates and comparision between CPU and GPU 
	// Do not forget to uncomment the chrono include
	//
  /* 
	cout << "===========GPU Ref time:===========" << endl;
  //printMatrix(a, n, n);

  auto begin = std::chrono::high_resolution_clock::now();
  
  auto end = std::chrono::high_resolution_clock::now();
  
  // cout << "============TIME==============" << endl;
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / float(1000000) << "ms" << std::endl;

  // check answer
  cout << "===========CPU time:===========" << endl;
  begin = std::chrono::high_resolution_clock::now();
  matrixMult(u, a, 1, n, n, n, res, 1, n);
  matrixMult(res, u, 1, n, n, 1, quad, 1, 1);
  
  end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / float(1000000) << "ms" << std::endl;
	*/
/*
  delete[] a;
  delete[] u;
  delete[] res;
  delete quad;

  return 0;
}*/
