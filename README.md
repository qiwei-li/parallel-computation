# parallel-computation
small projects of parallel computation with OpenMP, MPI, CUDA, OpenACC, and R

## authors
* Francois Demoullin
* Qiwei Li
* Erin Mcginnis
* Xiaotian Zhao

## files
* __nmfomp.cpp__

 Use OpenMP to perform nonnegative matrix factorization (NMF), using the multiplicative update method. NMF factorize matrix A (r x c) into W (r x k) and H (k x c). 
 
 Method used: Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." Advances in neural information processing systems. 2001.
 
 Compile: g++ -g -Wall -std=c++11 nmfomp.cpp -o a.out -fopenmp -lpthread

* __parquad.R__
 
 For a symmetric matrix A and a vector U, we wish to compute the quadratic form q = u'Au. This can be computed in parallel by partitioning matrix A properly. This code utilize the "Rdsm" package in R, which forces different R processes access the memory location of an object.

* __brightspot.cpp__

 Consider an n x n matrix of image pixels, with brightness values in [0,1]. Define a bright spot of size k and threshhold b to be a k x k subimage, with the pixels being contiguous and with each one having brightness at least b. The function returns a count of all bright spots using OpenMP
 
 Compile: g++ -g -Wall -std=c++11 brightspot.cpp -o a.out -fopenmp -lpthread

* __paths.cpp__

  Given a graph, find the count of all paths of a fix length and paths themselves using OpenMP. Loops are allowed, e.g. 5 → 2 → 0 → 2 → 0 → 88. The code utilizes a dynamic programming approach to find the vertices in a path given a starting point, parallelizing on different starting points.
    
  Compile: g++ -g -Wall -std=c++11 paths.cpp -o a.out -fopenmp -lpthread
  
* __nmfcuda.cu__

 Use CUDA to perform nonnegative matrix factorization (NMF).
 
 Compile: nvcc nmfcuda.cu -o a.out
 
* __gpuquad.cu__

 Use CUDA to compute the quadratic form q=u'Au.
 
 Compile: nvcc gpuquad.cu -o a.out
 
* __transgraph.cpp__

 Use MPI to tranform an adjacency matrix to a long format, where each row of the resulting matrix contains one pair of staring and ending vertices of a edge. 

 Compile: mpicxx -g -o a.out transgraph.cpp

 Run: mpiexec -f host3 -n 3 a.out

 Note that host3 should contain the names of all machines. You also need passwordless ssh login.
 
* __tutorial.pdf__

 This tutorial serves as an introduction to OpenACC. In this tutorial, we explain the basic use of OpenACC through many examples. All codes are contained.



