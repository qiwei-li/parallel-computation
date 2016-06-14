// g++ -g -Wall -std=c++11 nmfomp.cpp -o a.out -fopenmp -lpthread

#include <iostream>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <string.h>

using namespace std;

int MAX_NUM_THREADS = 8;

void matrixMultandEleWise(float* pOne, float* pTwo, int pOneR, int pOneC, int pTwoR, int pTwoC, float* pRes, int* pResR, int* pResC, float* pTop, float* pLeft)
{ 
  // check for consistency
  assert(pOneC == pTwoR);

  int lBlockSize = pOneR / MAX_NUM_THREADS;
  vector<int> lLowBounds;
  vector<int> lHighBounds;

  for(int i = 0; i < MAX_NUM_THREADS; i++)
  {
    lLowBounds.push_back(pOneR - (MAX_NUM_THREADS - i) * lBlockSize);
    lHighBounds.push_back( pOneR - (MAX_NUM_THREADS - i - 1)  * lBlockSize );
  }
  lLowBounds[0] = 0;
  lHighBounds[MAX_NUM_THREADS - 1] = pOneR;

  // outer dimensions are dimensions of resulting matrix
  *pResR = pOneR;
  *pResC = pTwoC;
  #pragma omp parallel num_threads(MAX_NUM_THREADS)
  {
    int th_id = omp_get_thread_num();
    // for each element in the resulting matrix
    for (int i = lLowBounds[th_id]; i < lHighBounds[th_id]; i++)
    {
      for (int j = 0; j < *pResC; j++)
      {
        float sum = 0;

        for (int k = 0; k < pOneC; k++)
        {
          sum += pOne[i * pOneC + k] * pTwo[ k * pTwoC + j];
        }

        // fill in the resulting matrix
        pRes[i * *pResC + j] = (pTop[i * *pResC + j] * pLeft[i * *pResC + j])/sum;
        
      } 
    }
  }
}

void matrixMult(float* pOne, float* pTwo, int pOneR, int pOneC, int pTwoR, int pTwoC, float* pRes, int* pResR, int* pResC)
{	
	// check for consistency
	assert(pOneC == pTwoR);
  
  //int fourth = pOneR/4;
  //int low_bounds[4] = {0 , pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth};
  //int high_bounds[4] = {pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth, pOneR};

  int lBlockSize = pOneR / MAX_NUM_THREADS;
  vector<int> lLowBounds;
  vector<int> lHighBounds;

  for(int i = 0; i < MAX_NUM_THREADS; i++)
  {
    lLowBounds.push_back(pOneR - (MAX_NUM_THREADS - i) * lBlockSize);
    lHighBounds.push_back( pOneR - (MAX_NUM_THREADS - i - 1)  * lBlockSize );
  }
  lLowBounds[0] = 0;
  lHighBounds[MAX_NUM_THREADS - 1] = pOneR;

  #pragma omp parallel num_threads(MAX_NUM_THREADS)
  {
    int th_id = omp_get_thread_num();
    // for each element in the resulting matrix
    for (int i = lLowBounds[th_id]; i < lHighBounds[th_id]; i++)
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
}

void elementMatrixMult(float* pOne, float* pTwo, int pOneR, int pOneC, int pTwoR, int pTwoC, float* pRes, int* pResR, int* pResC)
{
	// check for consistency
	assert(pOneC == pTwoC);
	assert(pOneR == pTwoR);

  //int fourth = pOneR/4;
  //int low_bounds[4] = {0 , pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth};
  //int high_bounds[4] = {pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth, pOneR};

  int lBlockSize = pOneR / MAX_NUM_THREADS;
  vector<int> lLowBounds;
  vector<int> lHighBounds;
  
  for(int i = 0; i < MAX_NUM_THREADS; i++)
  {
    lLowBounds.push_back(pOneR - (MAX_NUM_THREADS - i) * lBlockSize);
    lHighBounds.push_back( pOneR - (MAX_NUM_THREADS - i - 1)  * lBlockSize );
  }
  lLowBounds[0] = 0;
  lHighBounds[MAX_NUM_THREADS - 1] = pOneR;

  #pragma omp parallel num_threads(MAX_NUM_THREADS)
  {
    int th_id = omp_get_thread_num(); 
    for (int i = lLowBounds[th_id]; i < lHighBounds[th_id]; i++)
    {
      for (int j = 0; j < *pResC; j++)
      {
        // fill in the resulting matrix
        pRes[i * *pResC + j] = pOne[i * *pResC + j] * pTwo[i * *pResC + j];
      }	
    }
  }
}

void elementMatrixDiv(float* pOne, float* pTwo, int pOneR, int pOneC, int pTwoR, int pTwoC, float* pRes, int* pResR, int* pResC)
{
	// check for consistency
	assert(pOneC == pTwoC);
	assert(pOneR == pTwoR);

  //int fourth = pOneR/4;
  //int low_bounds[4] = {0 , pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth};
  //int high_bounds[4] = {pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth, pOneR};

  int lBlockSize = pOneR / MAX_NUM_THREADS;
  vector<int> lLowBounds;
  vector<int> lHighBounds;
  
  for(int i = 0; i < MAX_NUM_THREADS; i++)
  {
    lLowBounds.push_back(pOneR - (MAX_NUM_THREADS - i) * lBlockSize);
    lHighBounds.push_back( pOneR - (MAX_NUM_THREADS - i - 1)  * lBlockSize );
  }
  lLowBounds[0] = 0;
  lHighBounds[MAX_NUM_THREADS - 1] = pOneR;

  #pragma omp parallel num_threads(MAX_NUM_THREADS)
  {
    int th_id = omp_get_thread_num();
    for (int i = lLowBounds[th_id]; i < lHighBounds[th_id]; i++)
    {
      for (int j = 0; j < *pResC; j++)
      {
        // fill in the resulting matrix
        pRes[i * *pResC + j] = pOne[i * *pResC + j] / pTwo[i * *pResC + j];
      }	
    }
  }
}

void matrixTranspose(float* pOne, int pOneR, int pOneC, float* pRes, int* pResR, int* pResC)
{	
  
  //int fourth = pOneR/4;
  //int low_bounds[4] = {0 , pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth};
  //int high_bounds[4] = {pOneR - 3*fourth, pOneR - 2*fourth, pOneR - fourth, pOneR};

  int lBlockSize = pOneR / MAX_NUM_THREADS;
  vector<int> lLowBounds;
  vector<int> lHighBounds;
  
  for(int i = 0; i < MAX_NUM_THREADS; i++)
  {
    lLowBounds.push_back(pOneR - (MAX_NUM_THREADS - i) * lBlockSize);
    lHighBounds.push_back( pOneR - (MAX_NUM_THREADS - i - 1)  * lBlockSize );
  }
  lLowBounds[0] = 0;
  lHighBounds[MAX_NUM_THREADS - 1] = pOneR;

	// for each element in the resulting matrix
  #pragma omp parallel num_threads(MAX_NUM_THREADS)
  {
    int th_id = omp_get_thread_num();
    for (int i = lLowBounds[th_id]; i < lHighBounds[th_id]; i++)
	  {
      for (int j = 0; j < pOneC; j++)
      {
        pRes[j * *pResC + i] = pOne[i * pOneC + j];
      }	
	  }
  }
}

void printMatrix(float* mat, int limit, int col){
    // for large matrix
    // we only print the upper left corner
    for(int i = 0; i < limit; i++){
        for(int j = 0; j < limit; j++){
            cout << mat[i * col + j] << " ";
            if(j == limit - 1){
                cout << endl;
            }
        }
    }
}

void nmfomp(float* a, int r, int c, int k, int niters, float* w, float* h)
{
  // initialize W and H

  for(int i = 0; i < r * k; i++){
    w[i] = (float)rand()/(float)RAND_MAX;
  }

  for(int i = 0; i < k * c; i++){
    h[i] = (float)rand()/(float)RAND_MAX;
  }

  // cutoff, only use 8 threads if the matrix is really big
  // the rank seems to be irrelvant when deciding what MAX_NUM_THREADS should be.
  if (r * c >= 10000)
  {
    MAX_NUM_THREADS = 8;
  }
  else
  {
    MAX_NUM_THREADS = 4;
  }

	// dimensions of a: r x c
	// dimensions of w: r x k
	// dimensions of h: k x c

	float* lHTranspose = new float[k * c];
	int lHTransposeR = c;
	int lHTransposeC = k;

	float* lWTranspose = new float[r * k];
	int lWTransposeR = k;
	int lWTransposeC = r;
        
  // the following is needed for updating w
  // w_new = w * (A %*% H')/(W %*% H %*% H')
  // note * and / are elementwise
  // note %*% is regular matrix multiplication
  // 1st, A %*% H', r x k, store in lTop
  // 2nd, H %*% H', k x k, store in lTemp
  // 3rd, W %*% (H %*% H'), r x k, store in lBottom
	float* lTop = new float[r * k];
	int lTopR = r;
	int lTopC = k;

	float* lBottom = new float[r * k];
	int lBottomR = r;
	int lBottomC = k;

	float* lTemp = new float[k * k];
	int lTempR = k;
	int lTempC = k;

  // the following is needed for updating h
  // h_new = h * (W' %*% A)/(W' %*% W %*% H)
  // 1st, W' %*% A, k x c
  // 2nd, W' %*% W, k x k
  // 3rd, (W' %*% W) %*% H, k x c
  float* lTop2 = new float[k * c];
  int lTop2R = k;
  int lTop2C = c;

  float* lBottom2 = new float[k * c];
  int lBottom2R = k;
  int lBottom2C = c;

  float* lTemp2 = new float[k * k];
  int lTemp2R = k;
  int lTemp2C = k;

	for (int i = 0; i < niters; i++)
	{
		// update transposes
		matrixTranspose(h, k, c, lHTranspose, &lHTransposeR, &lHTransposeC);

		// update W

		// do the matrix multiplications on top and on the bottom of the fraction
		// A * H', stored in lTop
		matrixMult(a, lHTranspose, r, c, lHTransposeR, lHTransposeC, lTop, &lTopR, &lTopC);

		
    // H * H' , store result in temp
		matrixMult(h, lHTranspose, k, c, lHTransposeR, lHTransposeC, lTemp, &lTempR, &lTempC);
		
    matrixMultandEleWise(w, lTemp, r, k, lTempR, lTempC, lBottom, &lBottomR, &lBottomC, lTop, w);
    memcpy(w, lBottom, lBottomR * lBottomC * sizeof(float));


    // W * (HH'), stored in lBottom
		//matrixMult(w, lTemp, r, k, lTempR, lTempC, lBottom, &lBottomR, &lBottomC);
		// do the elementwise division, store result in lTop
		//elementMatrixDiv(lTop, lBottom, lTopR, lTopC, lBottomR, lBottomC, lTop, &lTopR, &lTopC);
		// do the elementwise multiplication, store result in w
		//elementMatrixMult(w, lTop, r, k, lTopR, lTopC, w, &r, &k);
  
    // update transposes
    matrixTranspose(w, r, k, lWTranspose, &lWTransposeR, &lWTransposeC);
		
    // update H

		// do the matrix multiplications on top and on the bottom of the fraction
		// W' * A
		matrixMult(lWTranspose, a, lWTransposeR, lWTransposeC, r, c, lTop2, &lTop2R, &lTop2C);

		// W' %*% W , store result in temp
		matrixMult(lWTranspose, w, lWTransposeR, lWTransposeC, r, k, lTemp2, &lTemp2R, &lTemp2C);
		
    matrixMultandEleWise(lTemp2, h, lTemp2R, lTemp2C, k, c, lBottom2, &lBottom2R, &lBottom2C, lTop2, h);
    memcpy(h, lBottom2, lBottom2R * lBottom2C * sizeof(float));


    // (W' %*% W) %*% H
		//matrixMult(lTemp2, h, lTemp2R, lTemp2C, k, c, lBottom2, &lBottom2R, &lBottom2C); 


		// do the elementwise division, store result in lTop
		//elementMatrixDiv(lTop2, lBottom2, lTop2R, lTop2C, lBottom2R, lBottom2C, lTop2, &lTop2R, &lTop2C);

		//elementMatrixMult(h, lTop2, k, c, lTop2R, lTop2C, h, &k, &c);

	}
}

int main()
{
	//debugging();

	// dimensions of a: r x c
	// dimensions of w: r x k
	// dimensions of h: k x c

	int lK = 130;

	int lAR = 200;
	int lAC = 210;
	float* lA = new float[lAR * lAC];

	int lHR = lK;
	int lHC = lAC;
	float* lH = new float[lHR * lHC];
	
	int lWR = lAR;
	int lWC = lK;
	float* lW = new float[lWR * lWC];


	for(int i = 0; i < lAR * lAC; i++){
		lA[i] = (float)rand()/(float)RAND_MAX;
	}
  
  // timing
  auto begin = std::chrono::high_resolution_clock::now();

	// here we go!!!!
	nmfomp(lA, lAR, lAC, lK, 100, lW, lH);

  auto end = std::chrono::high_resolution_clock::now();
  cout << "============TIME==============" << endl;
  cout << "Number of threads: " << MAX_NUM_THREADS << endl;
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / float(1000000) << "ms" << std::endl;

	// test bench:

	cout << "===========Original Matrix:===========" << endl;
	printMatrix(lA, 10, lAC);

	int lResR = lAR;
	int lResC = lAC;
	float* lRes = new float[lResR * lResC];

	matrixMult(lW, lH, lWR, lWC, lHR, lHC, lRes, &lResR, &lResC);

	cout << "===========Approximation:===========" << endl;
	printMatrix(lRes, 10, lResC);

	
  delete[] lA;
	delete[] lH;
	delete[] lW;
	delete[] lRes;

	return 0;	
}
