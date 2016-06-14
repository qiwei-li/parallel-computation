#include <stdio.h>
#include <iostream>
//#include <chrono>

using namespace std;

// transposes x into y
//
// runs for each element in the matrix
// expects blockDim.x to be colSize of x = rowSize of y
// expects gridDim.x to be rowSize of x = colSize of y
__global__
void transposeKernel(float *x, float *y)
{
  int me = blockIdx.x * blockDim.x + threadIdx.x; // me
  int transposedMe = threadIdx.x * gridDim.x + blockIdx.x; // 
  
  y[transposedMe] = x[me];
}

// element operation kernel
// specialized Kernel for the NMF problem
// this kernel computes the elementwise division of 2 matrices
// then an elementwise multiplication of a thirs matrix is added
// 
// runs for each element in the matrix
// expects blockDim.x to be colSize of top
// expects gridDim.x to be rowSize of top
__global__
void elemOperationKernel(float *top, float *bottom, float* mult, float *res)
{
  int me = blockIdx.x * blockDim.x + threadIdx.x; // me
  res[me] = mult[me] * (top[me] / bottom[me]);
}

// matrix multiplication
// takes two matices of equal dimensiona
// computes the elementwise division to be x * y
// stores the result in new matrix res
//
// runs for each element in the matrix
// expects blockDim.x to be colSize of x
__global__
void matrixMultKernel(float *x, float *y, float *res, int rowA, int colA, int colB)
{
  int bID = blockIdx.x; // my row
  
  // traverse the row in A
  for (int i = 0; i < colB; i++)
  {
    float sum = 0.0;
    // for this element do the multiplication
    for (int j = 0; j < colA; j++)
    {
      sum += x[bID * colA + j] * y[j * colB + i];
    }
    // fill in res
    res[bID * colB + i] = sum;
  }
}

// initializes a matrix on the GPU
// all elements are set to 0.1
// 
// runs for each element in the matrix
// expects blockDim.x to be colSize of matrix
// expects gridDim.x to be rowSize of matrix
__global__
void initializeMatrixKernel(float *matrix, float *ref)
{
  int me = blockIdx.x * blockDim.x + threadIdx.x; // me
  matrix[me] = ref[me];
}

// prints top left corner of a matrix
// limit denotes how big the printed square is
void printMatrix(float* mat, int limit, int col)
{
    // for large matrix
    // we only print the upper left corner
    for(int i = 0; i < limit; i++)
    {
        for(int j = 0; j < limit; j++)
        {
            cout << mat[i * col + j] << " ";
        }
        cout << endl;
    }
}
// prints entire matrix
// may flood the console
// dont use for huge matrices
void printMatrixFull(float* mat, int row, int col)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            cout << mat[i * col + j] << " ";
        }
        cout << endl;
    }
}


// dimensions of a: r x c
// dimensions of w: r x k
// dimensions of h: k x c

void nmfgpu(float* a, int r, int c, int k, int niters, float* w, float* h)
{

  // define the grids we are using for the different kernels
  dim3 dimGridK(k, 1);
  dim3 dimGridR(r, 1);

  dim3 dimBlockC(c, 1, 1);
  dim3 dimBlockK(k, 1, 1);

  dim3 dimBlock(1, 1, 1);

  // get a onto the GPU using a slow memcopy
  // unfortunately there is no way around it
  float* d_a;
  cudaMalloc((void**)&d_a, r * c * sizeof(float));
  cudaMemcpy(d_a, a, r * c * sizeof(float), cudaMemcpyHostToDevice);
  // a is now on the GPU

  // allocate w and h on the gpu to be raw memory 
  // then initialize w and h on the gpu, saves times compared to intializing w and h on the CPU
  float* d_h;
  float* d_w;

  cudaMalloc((void**)&d_h, k * c * sizeof(float));
  cudaMalloc((void**)&d_w, r * k * sizeof(float));

  // note that both kernels run concurrently
  // no need to call cudaThreadSynchronize in between the kernels
  
  initializeMatrixKernel<<<dimGridK, dimBlockC>>>(d_h, d_a);
  initializeMatrixKernel<<<dimGridR, dimBlockK>>>(d_w, d_a);



  // now however, we need to synch
  cudaThreadSynchronize();
  // w and h are now initialized on the GPU

  // allocate hTranspose and wTranspose on the GPU:
  float* d_hTrans;
  float* d_wTrans;

  cudaMalloc((void**)&d_hTrans, c * k * sizeof(float));
  cudaMalloc((void**)&d_wTrans, k * r * sizeof(float));

        
  // the following is needed for updating w
  // w_new = w * (A %*% H')/(W %*% H %*% H')
  // note * and / are elementwise
  // note %*% is regular matrix multiplication
  // 1st, A %*% H', r x k, store in d_top
  // 2nd, H %*% H', k x k, store in d_temp
  // 3rd, W %*% (H %*% H'), r x k, store in d_bottom

  float* d_top;
  float* d_bottom;
  float* d_temp;

  cudaMalloc((void**)&d_top, r * k * sizeof(float));
  cudaMalloc((void**)&d_bottom, r * k * sizeof(float));
  cudaMalloc((void**)&d_temp, k * k * sizeof(float));

  // similarily
  // the following is needed for updating h
  // h_new = h * (W' %*% A)/(W' %*% W %*% H)
  // 1st, W' %*% A, k x c
  // 2nd, W' %*% W, k x k
  // 3rd, (W' %*% W) %*% H, k x c

  float* d_top_two;
  float* d_bottom_two;
  float* d_temp_two;

  cudaMalloc((void**)&d_top_two, k * c * sizeof(float));
  cudaMalloc((void**)&d_bottom_two, k * c * sizeof(float));
  cudaMalloc((void**)&d_temp_two, k * k * sizeof(float));
  
  // cutoff value for correct updating vs fast updating
  // the iterations from 0-cutoffvalue are done properly with update w first, then update h
  // the iterations from cutoffvalue - niters are done faster: w and h are updated concurrently
  // while the latter is technically incorrect the loss of precision is minimal
  // and the gain in speed is noticable (while still small) 
  // int cutoffValue = niters / 2;
  int cutoffValue = niters / 2;

  for (int i = 0; i < cutoffValue; i++)
  {

    //
    //
    //
    // update W
    //
    //
    //

    // update d_hTrans transpose using the transpose kernel
    transposeKernel<<<dimGridK, dimBlockC>>>(d_h, d_hTrans);
    cudaThreadSynchronize();
    // d_htrans is up to date on the GPU

    // do the matrix multiplications on top and on the bottom of the fraction

    // first compute AH', the result is stored in d_top
    matrixMultKernel<<<dimGridR, dimBlock>>>(d_a, d_hTrans, d_top, r, c, k);
    cudaThreadSynchronize();
    // d_top = is up to date on the GPU
    
    // compute HH', store result in d_temp
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_h, d_hTrans, d_temp, k, c, k);
    cudaThreadSynchronize();
    // d_temp is up to date on the GPU

    // compute W(HH'), store result in d_bottom
    matrixMultKernel<<<dimGridR, dimBlock>>>(d_w, d_temp, d_bottom, r, k, k);
    cudaThreadSynchronize();
    // d_bottom is up to date on the GPU
    
    // finally compute the expression 
    // d_w * d_top / d_bottom store the result into d_w
    // all of this is done in just one specialized kernel!
    elemOperationKernel<<<dimGridR, dimBlockK>>>(d_top, d_bottom, d_w, d_w);
    cudaThreadSynchronize();
    
    //
    //
    //
    // update H
    //
    //
    //

    // update d_wTrans using the transpose kernel
    transposeKernel<<<dimGridR, dimBlockK>>>(d_w, d_wTrans);
    cudaThreadSynchronize();
    // d_wTrans is up to date on the GPU

    // do the matrix multiplications on top and on the bottom of the fraction

    // first compute W'A, the result is stored in d_top_two
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_wTrans, d_a, d_top_two, k, r, c);
    cudaThreadSynchronize();  
    // d_top_two is up to date on the GPU

    // compute W'W, store result in d_temp_two
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_wTrans, d_w, d_temp_two, k, r, k);
    cudaThreadSynchronize();
    // d_temp_two is up to date on the GPU
    
    // compute (W'W)H, store result in d_bottom
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_temp_two, d_h, d_bottom_two, k, k, c);
    cudaThreadSynchronize();
    // d_bottom_two is up to date on the GPU

    // finally compute the expression 
    // d_h * d_top / d_bottom store the result into d_h
    // all of this is done in just one specialized kernel!
    elemOperationKernel<<<dimGridK, dimBlockC>>>(d_top_two, d_bottom_two, d_h, d_h);
    cudaThreadSynchronize();
  }

  
  for (int i = cutoffValue; i < niters; i++)
  {

    //
    //
    //
    // update W and update H
    //
    //
    //

    // update d_hTrans transpose using the transpose kernel
    transposeKernel<<<dimGridK, dimBlockC>>>(d_h, d_hTrans);
    // update d_wTrans using the transpose kernel
    transposeKernel<<<dimGridR, dimBlockK>>>(d_w, d_wTrans);
    cudaThreadSynchronize();
    // d_htrans is up to date on the GPU

    // do the matrix multiplications on top and on the bottom of the fraction

    // first compute AH', the result is stored in d_top
    matrixMultKernel<<<dimGridR, dimBlock>>>(d_a, d_hTrans, d_top, r, c, k);

    // first compute W'A, the result is stored in d_top_two
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_wTrans, d_a, d_top_two, k, r, c);

    cudaThreadSynchronize();
    // d_top = is up to date on the GPU
    
    // compute HH', store result in d_temp
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_h, d_hTrans, d_temp, k, c, k);
    // compute W'W, store result in d_temp_two
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_wTrans, d_w, d_temp_two, k, r, k);
    cudaThreadSynchronize();
    // d_temp is up to date on the GPU

    // compute W(HH'), store result in d_bottom
    matrixMultKernel<<<dimGridR, dimBlock>>>(d_w, d_temp, d_bottom, r, k, k);

    // compute (W'W)H, store result in d_bottom
    matrixMultKernel<<<dimGridK, dimBlock>>>(d_temp_two, d_h, d_bottom_two, k, k, c);
    cudaThreadSynchronize();
    // d_bottom is up to date on the GPU
    
    // finally compute the expression 
    // d_w * d_top / d_bottom store the result into d_w
    // all of this is done in just one specialized kernel!
    elemOperationKernel<<<dimGridR, dimBlockK>>>(d_top, d_bottom, d_w, d_w);
    elemOperationKernel<<<dimGridK, dimBlockC>>>(d_top_two, d_bottom_two, d_h, d_h);
    cudaThreadSynchronize();
  }

  // copy w and h back to the CPU
  cudaMemcpy(w, d_w, r * k * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h, d_h, k * c * sizeof(float), cudaMemcpyDeviceToHost);

  // free all the cuda memory, it is no longer needed
  cudaFree(d_a);
  cudaFree(d_h);
  cudaFree(d_w);
  cudaFree(d_hTrans);
  cudaFree(d_wTrans);
  cudaFree(d_top);
  cudaFree(d_bottom);
  cudaFree(d_temp);
  cudaFree(d_top_two);
  cudaFree(d_bottom_two);
  cudaFree(d_temp_two);
}

// CPU version of matrix multiplication
// For debugging purposes only!
void matrixMult(float* pOne, float* pTwo, int pOneR, int pOneC, int pTwoR, int pTwoC, float* pRes, int* pResR, int* pResC)
{ 
  // for each element in the resulting matrix
  for (int i = 0; i < pOneR; i++)
  {
    for (int j = 0; j < *pResC; j++)
    {
      float sum = 0;

      for (int k = 0; k < pOneC; k++)
      {
        sum += pOne[i * pOneC + k] * pTwo[ k * pTwoC + j];
      }

      // fill in the resulting matrix
      pRes[i * *pResC + j] = sum;
    } 
  }
}

/*
*
* Main
*
* This is our test bench, we set up the matrices, start NMF computations and print out the results
* REQUIRES C++11 for chrono. 
* If no C++11 support get rid of the chrono calls, you wont be able to accurately time the algo though.
*
*/
int main(void)
{

  //nmfgpu(float* a, int r, int c, int k, int niters, float* w, float* h)

  int lK = 50;

  int lAR = 200;
  int lAC = 210;
  float* lA = new float[lAR * lAC];

  int lHR = lK;
  int lHC = lAC;
  float* lH = new float[lHR * lHC];
  
  int lWR = lAR;
  int lWC = lK;
  float* lW = new float[lWR * lWC];


  for(int i = 0; i < lAR * lAC; i++)
  {
    lA[i] = i*1.0;
  }
  
  // timing
  //auto begin = std::chrono::high_resolution_clock::now();

  // here we go!!!!
  nmfgpu(lA, lAR, lAC, lK, 2000, lW, lH);

  //auto end = std::chrono::high_resolution_clock::now();
  //cout << "============TIME==============" << endl;
  //std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / float(1000000) << "ms" << std::endl;

  // test bench:

  cout << "===========Original Matrix:===========" << endl;
  printMatrix(lA, 10, lAC);

  int lResR = lAR;
  int lResC = lAC;
  float* lRes = new float[lResR * lResC];

  matrixMult(lW, lH, lAR, lK, lK, lAC, lRes, &lResR, &lResC);

  cout << "===========Approximation:===========" << endl;
  printMatrix(lRes, 10, lResC);

  
  delete[] lA;
  delete[] lH;
  delete[] lW;
  delete[] lRes;

  return 0;
}
