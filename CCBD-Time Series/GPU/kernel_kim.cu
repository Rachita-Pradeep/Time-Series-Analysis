#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))
#define dist(x,y) ((x-y)*(x-y))

#define INF 1e10       //Pseudo Infitinte number for this code

/// Calculate quick lower bound
/// Usually, LB_Kim take time O(m) for finding top,bottom,fist and last.
/// However, because of z-normalization the top and bottom cannot give siginifant benefits.
/// And using the first and last points can be computed in constant time.
/// The prunning power of LB_Kim is non-trivial, especially when the query is not long, say in length 128.
/////////////////////Added to use constant memory///////////////
//extern __constant__ double q[];
//////////////////////////////////////////////////////
///////////////////// Added to include locs  ///////////////////
__device__ int lock_Variable = 0; //0 loc open, 1 closed
/////////////////////////////////////
__device__ double lb_kim_hierarchy(double *t, double *q, int j, int len,
		double mean, double std, double bsf = INF) {
	/// 1 point at front and back
	double d, lb;
	double x0 = (t[j] - mean) / std;
	double y0 = (t[(len - 1 + j)] - mean) / std;
	lb = dist(x0,q[0]) + dist(y0,q[len-1]);
	if (lb >= bsf)
		return lb;

	/// 2 points at front
	double x1 = (t[(j + 1)] - mean) / std;
	d = min(dist(x1,q[0]), dist(x0,q[1]));
	d = min(d, dist(x1,q[1]));
	lb += d;
	if (lb >= bsf)
		return lb;

	/// 2 points at back
	double y1 = (t[(len - 2 + j)] - mean) / std;
	d = min(dist(y1,q[len-1]), dist(y0, q[len-2]) );
	d = min(d, dist(y1,q[len-2]));
	lb += d;
	if (lb >= bsf)
		return lb;

	/// 3 points at front
	double x2 = (t[(j + 2)] - mean) / std;
	d = min(dist(x0,q[2]), dist(x1, q[2]));
	d = min(d, dist(x2,q[2]));
	d = min(d, dist(x2,q[1]));
	d = min(d, dist(x2,q[0]));
	lb += d;
	if (lb >= bsf)
		return lb;

	/// 3 points at back
	double y2 = (t[(len - 3 + j)] - mean) / std;
	d = min(dist(y0,q[len-3]), dist(y1, q[len-3]));
	d = min(d, dist(y2,q[len-3]));
	d = min(d, dist(y2,q[len-2]));
	d = min(d, dist(y2,q[len-1]));
	lb += d;

	return lb;
}

__device__ double dtw(double* A, double* B, int m, int r, double* costM,
		double* cost_prevM, int bsfindex, double bsf = INF) {

	double *cost_tmp;
	int i, j, k;
	double x, y, z, min_cost;
	int start = bsfindex * (2 * r + 1);
	double* cost = costM + start;
	double*cost_prev = cost_prevM + start;

	/// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
//	cudaMalloc((void**)&cost, (2*r+1) * sizeof(double));
	for (k = 0; k < 2 * r + 1; k++)
		cost[k] = INF;

//	cudaMalloc((void**)&cost_prev, (2*r+1) * sizeof(double));
	for (k = 0; k < 2 * r + 1; k++)
		cost_prev[k] = INF;

	for (i = 0; i < m; i++) {
		k = max(0,r-i);
		min_cost = INF;

		for (j = max(0,i-r); j <= min(m-1,i+r); j++, k++) {
			/// Initialize all row and column
			if ((i == 0) && (j == 0)) {
				cost[k] = dist(A[0],B[0]);
				min_cost = cost[k];
				continue;
			}

			if ((j - 1 < 0) || (k - 1 < 0))
				y = INF;
			else
				y = cost[k - 1];
			if ((i - 1 < 0) || (k + 1 > 2 * r))
				x = INF;
			else
				x = cost_prev[k + 1];
			if ((i - 1 < 0) || (j - 1 < 0))
				z = INF;
			else
				z = cost_prev[k];

			/// Classic DTW calculation
			cost[k] = min( min( x, y) , z) + dist(A[i],B[j]);

			/// Find minimum cost in row for early abandoning (possibly to use column instead of row).
			if (cost[k] < min_cost) {
				min_cost = cost[k];
			}
		}

		/// We can abandon early if the current cummulative distace with lower bound together are larger than bsf
		if (i + r < m - 1 && min_cost >= bsf) {
			return min_cost;
		}

		/// Move current array to previous array.
		cost_tmp = cost;
		cost = cost_prev;
		cost_prev = cost_tmp;
	}
	k--;

	/// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
	double final_dtw = cost_prev[k];

	return final_dtw;
}

__global__ void processKernel(double* queue, double* buffer, double* cost,
		double* cost_prev, double* bsf_a, int* loc_a, double* tM, double* tzM,
		int m, int r, double bsf, int size, int EPOCH) {

	extern __shared__ double q[];
	int shared_index = threadIdx.x;
	while(shared_index < m){
		q[shared_index]  = queue[shared_index];
	shared_index += blockDim.x;
	}

	//printf("Hello");
	int N = gridDim.x;
	int M = blockDim.x;
	int i = blockIdx.x;
	int j = threadIdx.x;

	int items_per_a = EPOCH / (N * M);
	int maxindex = (size - 1) / items_per_a;
	double lb_kim;
	int bsfindex = i * M + j;
	int sindex = bsfindex * items_per_a;
	int loc;
	int k;
	double d;
	double *t, *tz;
	double ex, ex2, mean, std, dist;
	t = tM + bsfindex * 2 * m;
	tz = tzM + bsfindex * 2 * m;
	/// Initial the cummulative lower bound
	ex = 0;
	ex2 = 0;
	int offset = m;
	if (bsfindex == maxindex)
		offset = 0;
	if (bsfindex <= maxindex)
		for (i = 0; i < items_per_a + offset; i++) {
			d = (double) buffer[sindex + i];
			ex += d;
			ex2 += d * d;
			t[i % m] = d;
			t[(i % m) + m] = d;

			/// If there is enough data in t, the DTW distance can be calculated
			if (i >= m - 1) {
				mean = ex / m;
				std = ex2 / m;
				std = sqrt(std - mean * mean);

				/// compute the start location of the data in the current circular array, t
				j = (i + 1) % m;

				/// Use a constant lower bound to prune the obvious subsequence
				lb_kim = lb_kim_hierarchy(t, q, j, m, mean, std, bsf);

				if (lb_kim < bsf)
				{

					for (k = 0; k < m; k++) {
						tz[k] = (t[(k + j)] - mean) / std;
					}

					dist = dtw(tz, q, m, r, cost, cost_prev, bsfindex, bsf);

					////////////////////////////// Implementing locks //////////////////////////////////
					///Previous code
					//
					//	if (dist < bsf) {   /// Update bsf
					//				/// loc is the real starting location of the nearest neighbor in the file
					//		bsf = dist;
					//		loc = sindex + i;
					//	}
					/////////End of previous code

			 		///////// Implementing loc
					if (dist < bsf) {
						bool loop = true;
						while (loop) {
							if (atomicCAS(&lock_Variable, 0, 1)) { //If loc open (loc == 0) then close it (make it equal to 1)
								if (dist < bsf) {
									bsf = dist;
									loc = sindex + i;
								}
						 		lock_Variable = 0;
								loop = false;
							}
						}
					}
					///////////////////////////////////////////////////////////////////////////////////
				}
				/// Reduce obsolute points from sum and sum square
				ex -= t[j];
				ex2 -= t[j] * t[j];
			}
		}
	bsf_a[bsfindex] = bsf;
	loc_a[bsfindex] = loc;
	//Some issue which popped up now .

}

void error(int id) {
	if (id == 1)
		printf("ERROR : Memory can't be allocated!!!\n\n");
	else if (id == 2)
		printf("ERROR : File not Found!!!\n\n");
	else if (id == 3)
		printf("ERROR : Can't create Output File!!!\n\n");
	else if (id == 4) {
		printf("ERROR : Invalid Number of Arguments!!!\n");
		printf(
				"Command Usage:  UCR_DTW.exe  data-file  query-file   m   R\n\n");
		printf(
				"For example  :  UCR_DTW.exe  data.txt   query.txt   128  0.05\n");
	}
	exit(1);
}

/// Main Function
int main(int argc, char *argv[]) {
	FILE *fp; /// data file pointer
	FILE *qp; /// query file pointer
	double bsf = INF; /// best-so-far
	double *h_q; /// data array and query array
	clock_t begin, end;
	double time_spent;

	double d;
	long long i;
	double ex, ex2, mean, std;
	int m = -1, r = -1;
	long long loc = 0;
	double t1, t2;
	int kim = 0, keogh = 0, keogh2 = 0;
	double *h_buffer;
	int N = 10, M = 100;
	int sh= 0;

	/// For every EPOCH points, all cummulative values, such as ex (sum), ex2 (sum square), will be restarted for reducing the doubleing point error.
	int EPOCH = 1000000;
	int epoch; //Optimization
	/// If not enough input, display an error.

	if (argc <= 3)
		error(4);

	/// read size of the query
	if (argc > 3)
		m = atol(argv[3]);

	/// read warping windows
	if (argc > 4) {
		double R = atof(argv[4]);
		if (R <= 1)
			r = floor(R * m);
		else
			r = floor(R);
	}
	if (argc > 7) {
		N = atoi(argv[5]);
		M = atoi(argv[6]);
		EPOCH = atol(argv[7]);
	}
//	m = 128;
//	r = 6;
	fp = fopen(argv[1], "r");
//	fp = fopen("/home/ubuntu/Desktop/DTW Project/Executable/Data.txt", "r");
	// if( fp == NULL )
	//     error(2);

	qp = fopen(argv[2], "r");
//	qp = fopen("/home/ubuntu/Desktop/DTW Project/Executable/Query.txt", "r");
	//  if( qp == NULL )
	//     error(2);

	/// start the clock
	t1 = clock();

	/// malloc everything here
	h_q = (double *) malloc(sizeof(double) * m);
	if (h_q == NULL)
		error(1);

	h_buffer = (double *) malloc(sizeof(double) * (EPOCH));
	if (h_buffer == NULL)
		error(1);

	/// Read query file
	bsf = INF;
	i = 0;
	ex = ex2 = 0;

	while (fscanf(qp, "%lf", &d) != EOF && i < m) {
		ex += d;
		ex2 += d * d;
		h_q[i] = d;
		i++;
	}
	fclose(qp);

	/// Do z-normalize the query, keep in same array, q
	mean = ex / m;
	std = ex2 / m;
	std = sqrt(std - mean * mean);
	for (i = 0; i < m; i++)
		h_q[i] = (h_q[i] - mean) / std;

	int size = N * M;
	double* h_bsf = (double *) malloc(sizeof(double) * size);
	int* h_loc = (int *) malloc(sizeof(int) * size);
	for (i = 0; i < size; i++) {
		h_bsf[i] = INF;
		h_loc[i] = 0;
	}
	//Allocate all the cuda Stuffs
	double *d_q;
	double *d_buffer, *d_bsf;
	double *d_cost, *d_cost_prev;
	double *d_t, *d_tz;
	int* d_loc;

	cudaMalloc((void**) &d_buffer, (EPOCH) * sizeof(double));
	cudaMalloc((void**) &d_cost, (2 * r + 1) * size * sizeof(double));
	cudaMalloc((void**) &d_cost_prev, (2 * r + 1) * size * sizeof(double));
	cudaMalloc((void**) &d_bsf, size * sizeof(double));
	cudaMalloc((void**) &d_t, 2 * m * size * sizeof(double));
	cudaMalloc((void**) &d_tz, 2 * m * size * sizeof(double));

	cudaMalloc((void**) &d_q, m * sizeof(double));

	cudaMalloc((void**) &d_loc, size * sizeof(int));

	///Copying BSF array
	cudaMemcpy(d_bsf, h_bsf, m * sizeof(double), cudaMemcpyHostToDevice);

	///Copy all the Query related arrays
	cudaMemcpy(d_q, h_q, m * sizeof(double), cudaMemcpyHostToDevice);
	bool done = false;
	bool last = false;
	int it = 0, ep = 0, k = 0;
	//begin = clock();
	while (!done) {
		/// Read first m-1 points
		if (it == 0) {
			epoch = 100000;
			while (ep < epoch) {
				if (fscanf(fp, "%lf", &d) == EOF)
					break;
				h_buffer[ep] = d;
				ep++;
			}
		}

		/// Data are read in chunk of size EPOCH.
		/// When there is nothing to read, the loop is end.
		if (ep <= m - 1) {
			done = true;
		} else {
			if (last) {
				done = true;
			}
			//printf("Reading Done.\n");
			sh ++;
			//begin = clock();
			cudaMemcpy(d_buffer, h_buffer, ep * sizeof(double),
					cudaMemcpyHostToDevice); // to copy from CPU to GPU
			cudaDeviceSynchronize();
		    //end = clock();
		   // time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
		  //  printf("Time taken by memcpy for reading buffer %lf ", time_spent);
			/// Just for printing a dot for approximate a million point. Not much accurate.
//			printf("Copying done.\n");
			//Do everything here
		//	begin = clock();
			processKernel<<<N, M,m*sizeof(double)>>>(d_q, d_buffer, d_cost, d_cost_prev, d_bsf,
					d_loc, d_t, d_tz, m, r, bsf, ep, EPOCH);
		//	cudaDeviceSynchronize();
      //      end = clock();
    //        time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
  //          printf("Time taken by kernel %lf ", time_spent);
			//Do the next set of buffering

//		printf("Kernel done.\n");
			epoch = EPOCH;
			ep = 0;
//			begin = clock();
			while (ep < epoch) {
				if (fscanf(fp, "%lf", &d) == EOF) {
					last = true;
					break;
				}
				h_buffer[ep] = d;
				ep++;
			}
//			end = clock();
//			time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
//		printf("Time taken for reading %lf ", time_spent);
		
		//printf("Loading next set done\n");
//		begin = clock();
			cudaMemcpy(h_bsf, d_bsf, size * sizeof(double),
					cudaMemcpyDeviceToHost);

			cudaMemcpy(h_loc, d_loc, size * sizeof(int),
					cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
//			end = clock();
//			time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
//				   printf("Time taken for memcpy %lf", time_spent);
			//	   printf("computation");
			begin = clock();
			for (k = 0; k < size; k++) {
				if (bsf > h_bsf[k]) {
					bsf = h_bsf[k];
					if (it == 0) {
						loc = (it) * (EPOCH) + h_loc[k] - m + 1;
					} else {
						loc = 100000 + (it - 1) * (EPOCH) + h_loc[k] - m + 1;
					}
				}
			}
//			end = clock();
//			time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
//							   printf("Time taken for computation %lf \n", time_spent);
		//	printf("Computation Done.\n");


			/// If the size of last chunk is less then EPOCH, then no more data and terminate.

		}
		it++;
	}
//	end = clock();
//	time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
//	printf("\nTime taken %lf ", time_spent);
	fclose(fp);

	free(h_q);
	free(h_buffer);
	free(h_bsf);

	cudaFree(d_buffer);
	cudaFree(d_q);
	cudaFree(d_bsf);
	cudaFree(d_loc);
	cudaFree(d_cost);
	cudaFree(d_cost_prev);
	cudaFree(d_t);
	cudaFree(d_tz);

//	t2 = clock();
	printf("\n");

	/// Note that loc and i are long long.
	//   cout << "Location : " << loc << endl;
	//  cout << "Distance : " << sqrt(bsf) << endl;
//    cout << "Data Scanned : " << i << endl;
	//   cout << "Total Execution Time : " << (t2-t1)/CLOCKS_PER_SEC << " sec" << endl;

	/// printf is just easier for formating ;)
	printf("Distance  %lf\n", sqrt(bsf));
	printf("Location %d\n", loc);
	printf("No of iterations %d\n", sh);
	return 0;
}
