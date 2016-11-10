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

__device__ int lock_Variable = 0; //0 loc open, 1 closed

// holds results of kim
struct kim_results
{
	int loc;
	double chunk[128];	
};

// holds results of keogh
struct keogh_results
{
	int loc;
	double chunk[128];	
};

/// Data structure for sorting the query
typedef struct Index
{
	double value;
	int index;
} Index;

/// Data structure (circular array) for finding minimum and maximum for LB_Keogh envolop
struct deque
{
	int *dq;
	int size, capacity;
	int f, r;
};

/// Sorting function for the query, sort by abs(z_norm(q[i])) from high to low
int comp(const void *a, const void* b) {
	Index* x = (Index*) a;
	Index* y = (Index*) b;
	int aa = (y->value) < 0 ? -(y->value) : y->value;
	int bb = (x->value) < 0 ? -(x->value) : x->value;
	return aa - bb;
}

/// Initial the queue at the begining step of envelop calculation
void init(deque *d, int capacity) {
	d->capacity = capacity;
	d->size = 0;
	d->dq = (int *) malloc(sizeof(int) * d->capacity);
	d->f = 0;
	d->r = d->capacity - 1;
}

/// Destroy the queue
void destroy(deque *d) {
	free(d->dq);
}

/// Insert to the queue at the back
void push_back(struct deque *d, int v) {
	d->dq[d->r] = v;
	d->r--;
	if (d->r < 0)
		d->r = d->capacity - 1;
	d->size++;
}

/// Delete the current (front) element from queue
void pop_front(struct deque *d) {
	d->f--;
	if (d->f < 0)
		d->f = d->capacity - 1;
	d->size--;
}

/// Delete the last element from queue
void pop_back(struct deque *d) {
	d->r = (d->r + 1) % d->capacity;
	d->size--;
}

/// Get the value at the current position of the circular queue
int front(struct deque *d) {
	int aux = d->f - 1;

	if (aux < 0)
		aux = d->capacity - 1;
	return d->dq[aux];
}

/// Get the value at the last position of the circular queueint back(struct deque *d)
int back(struct deque *d) {
	int aux = (d->r + 1) % d->capacity;
	return d->dq[aux];
}

/// Check whether or not the queue is empty
int empty(struct deque *d) {
	return d->size == 0;
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
		printf("Command Usage:  UCR_DTW.exe  data-file  query-file   m   R\n\n");
		printf("For example  :  UCR_DTW.exe  data.txt   query.txt   128  0.05\n");
	}
	exit(1);
}

/// Finding the envelop of min and max value for LB_Keogh
/// Implementation idea is intoruduced by Danial Lemire in his paper
/// "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound", Pattern Recognition 42(9), 2009.
void lower_upper_lemire(double *t, int len, int r, double *l, double *u) {
	struct deque du, dl;

	init(&du, 2 * r + 2);
	init(&dl, 2 * r + 2);

	push_back(&du, 0);
	push_back(&dl, 0);

	for (int i = 1; i < len; i++) {
		if (i > r) {
			u[i - r - 1] = t[front(&du)];
			l[i - r - 1] = t[front(&dl)];
		}
		if (t[i] > t[i - 1]) {
			pop_back(&du);
			while (!empty(&du) && t[i] > t[back(&du)])
				pop_back(&du);
		} else {
			pop_back(&dl);
			while (!empty(&dl) && t[i] < t[back(&dl)])
				pop_back(&dl);
		}
		push_back(&du, i);
		push_back(&dl, i);
		if (i == 2 * r + 1 + front(&du))
			pop_front(&du);
		else if (i == 2 * r + 1 + front(&dl))
			pop_front(&dl);
	}
	for (int i = len; i < len + r + 1; i++) {
		u[i - r - 1] = t[front(&du)];
		l[i - r - 1] = t[front(&dl)];
		if (i - front(&du) >= 2 * r + 1)
			pop_front(&du);
		if (i - front(&dl) >= 2 * r + 1)
			pop_front(&dl);
	}
	destroy(&du);
	destroy(&dl);
}

/// LB_Keogh 1: Create Envelop for the query
/// Note that because the query is known, envelop can be created once at the begenining.

/// Variable Explanation,
/// order : sorted indices for the query.
/// uo, lo: upper and lower envelops for the query, which already sorted.
/// t     : a circular array keeping the current data.
/// j     : index of the starting location in t
/// cb    : (output) current bound at each position. It will be used later for early abandoning in DTW.
__device__ void lb_keogh_cumulative(const int* order, const double *t,
		const double *uo, const double *lo, double *cb, int j, int len,
		double mean, double std, double best_so_far = INF) 
{
	double lb = 0;
	double x, d;

	for (int i = 0; i < len && lb < best_so_far; i++) {
		x = (t[(order[i] + j)] - mean) / std;
		d = 0;
		if (x > uo[i])
			d = dist(x,uo[i]);
		else if (x < lo[i])
			d = dist(x,lo[i]);
		lb += d;
		//	cb[order[i]] = d;
	}

	if(lb < bsf)
	{
		// add into the queue		
	}
}

/// Calculate quick lower bound
/// Usually, LB_Kim take time O(m) for finding top,bottom,fist and last.
/// However, because of z-normalization the top and bottom cannot give siginifant benefits.
/// And using the first and last points can be computed in constant time.
/// The prunning power of LB_Kim is non-trivial, especially when the query is not long, say in length 128.
__device__ void lb_kim_hierarchy(double *t, double *q, int j, int len,
		double mean, double std, double bsf = INF) 
{
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

	if(lb < bsf)
	{
		// add into the queue
	}
}

__device__ double dtw(double* A, double* B, int m, int r, double* costM,
		double* cost_prevM, int bsfindex,
		double * cb,
		double bsf = INF) 
{
	double *cost_tmp;
	int i, j, k;
	double x, y, z, min_cost;
	int start = bsfindex * (2 * r + 1);
	double* cost = costM + start;
	double*cost_prev = cost_prevM + start;

	/// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
	for (k = 0; k < 2 * r + 1; k++)
		cost[k] = INF;

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
		if (i + r < m - 1 && min_cost + cb[i + r + 1] >= bsf) {
			return min_cost + cb[i + r + 1];
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

__global__ void initKernel(double* queue, int m) 
{
	extern __shared__ double q[];
	int shared_index = threadIdx.x;
	while (shared_index < m) {
		q[shared_index] = queue[shared_index];
		shared_index += blockDim.x;
	}
}

__global__ void processKernel(double* queue, double* buffer, double* cost,
		double* cost_prev, double* bsf_a, int* loc_a, double* tM, double* tzM,
		int m, int r, double bsf, int size, int EPOCH,
		int * order, double * uo, double * lo, double * cbFull) 
{
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
	double * cb = cbFull + bsfindex * m;
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
				//lb_kim = 
				// launch a kim kernel
				lb_kim_hierarchy(t, q, j, m, mean, std, bsf);

				// launch a keogh kernel
				if (lb_kim < bsf) {
					double lb_k; // = lb_keogh_cumulative(order, t, uo, lo, cb, j,
					//m, mean, std, bsf);

					double lb = 0;
					double x, d;
					for (int ii = 0; ii < m && lb < bsf; ii++) {
						x = t[(order[ii] + j)];
						x = x - mean;
						double y = 1 / std;
						x = x * y;
						d = 0;
						if (x > uo[ii])
							d = dist(x,uo[ii]);
						else if (x < lo[ii])
							d = dist(x,lo[ii]);
						lb += d;
						cb[order[ii]] = d;
					}
					lb_k = lb;

					if (lb_k < bsf) {
						for (k = 0; k < m; k++) {
							tz[k] = (t[(k + j)] - mean) / std;
						}

						dist = dtw(tz, q, m, r, cost, cost_prev, bsfindex, cb,
								bsf);

						// Implementing loc
						if (dist < bsf) {
							bool loop = true;
							while (loop) {
								if (atomicCAS(&lock_Variable, 0, 1)) 
								{ 
									// If loc open (loc == 0) then close it (make it equal to 1)
									if (dist < bsf) {
										bsf = dist;
										loc = sindex + i;
									}
									lock_Variable = 0;
									loop = false;
								}
							}
						}
					}
				}
				/// Reduce obsolute points from sum and sum square
				ex -= t[j];
				ex2 -= t[j] * t[j];
			}
		}
	bsf_a[bsfindex] = bsf;
	loc_a[bsfindex] = loc;
}

/// Main Function
int main(int argc, char *argv[]) {
	FILE *fp; /// data file pointer
	FILE *qp; /// query file pointer
	double bsf = INF; /// best-so-far
	double *h_q; /// data array and query array

	double d;
	long long i;
	double ex, ex2, mean, std;
	int m = -1, r = -1;
	long long loc = 0;
	double *h_buffer;
	int N = 30, M = 192;
	int sh = 0;

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
	fp = fopen(argv[1], "r");
	if( fp == NULL )
	    error(2);

	qp = fopen(argv[2], "r");
	 if( qp == NULL)
	    error(2);

	/// malloc everything here
	h_q = (double *) malloc(sizeof(double) * m);
	if (h_q == NULL)
		error(1);

	h_buffer = (double *) malloc(sizeof(double) * (EPOCH));
	if (h_buffer == NULL)
		error(1);

	double * uo = (double *) malloc(sizeof(double) * m);
	if (uo == NULL)
		error(1);
	double * lo = (double *) malloc(sizeof(double) * m);
	if (lo == NULL)
		error(1);

	int * order = (int *) malloc(sizeof(int) * m);
	if (order == NULL)
		error(1);

	Index * Q_tmp = (Index *) malloc(sizeof(Index) * m);
	if (Q_tmp == NULL)
		error(1);

	double * u = (double *) malloc(sizeof(double) * m);
	if (u == NULL)
		error(1);

	double * l = (double *) malloc(sizeof(double) * m);
	if (l == NULL)
		error(1);

	double * cb1 = (double *) calloc(M * N * m, sizeof(double));
	if (cb1 == NULL)
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

	/// Create envelop of the query: lower envelop, l, and upper envelop, u
	lower_upper_lemire(h_q, m, r, l, u);

	/// Sort the query one time by abs(z-norm(q[i]))
	for (i = 0; i < m; i++) {
		Q_tmp[i].value = h_q[i];
		Q_tmp[i].index = i;
	}
	qsort(Q_tmp, m, sizeof(Index), comp);

	/// also create another arrays for keeping sorted envelop
	for (i = 0; i < m; i++) {
		int o = Q_tmp[i].index;
		order[i] = o;
		uo[i] = u[o];
		lo[i] = l[o];
	}

	free(Q_tmp);

	int size = N * M;
	double* h_bsf = (double *) malloc(sizeof(double) * size);
	int* h_loc = (int *) malloc(sizeof(int) * size);
	for (i = 0; i < size; i++) {
		h_bsf[i] = INF;
		h_loc[i] = 0;
	}

	// Allocate all the cuda Stuffs
	double *d_q;
	double *d_buffer, *d_bsf;
	double *d_cost, *d_cost_prev;
	double *d_t, *d_tz;
	int* d_loc;

	int * d_order;
	double *d_uo;
	double *d_lo;
	double *d_cb1;

	cudaMalloc((void**) &d_uo, sizeof(double) * m);
	cudaMalloc((void**) &d_lo, sizeof(double) * m);
	cudaMalloc((void**) &d_order, sizeof(int) * m);
	cudaMalloc((void**) &d_cb1, sizeof(double) * M * N * m);

	cudaMalloc((void**) &d_buffer, (EPOCH) * sizeof(double));
	cudaMalloc((void**) &d_cost, (2 * r + 1) * size * sizeof(double));
	cudaMalloc((void**) &d_cost_prev, (2 * r + 1) * size * sizeof(double));
	cudaMalloc((void**) &d_bsf, size * sizeof(double));
	cudaMalloc((void**) &d_t, 2 * m * size * sizeof(double));
	cudaMalloc((void**) &d_tz, 2 * m * size * sizeof(double));

	cudaMalloc((void**) &d_q, m * sizeof(double));

	cudaMalloc((void**) &d_loc, size * sizeof(int));
	// Added for lb keogh
	cudaMemcpy(d_lo, lo, m * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_uo, uo, m * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_order, order, m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cb1, cb1, M * N * m * sizeof(double), cudaMemcpyHostToDevice);

	// Copying BSF array
	cudaMemcpy(d_bsf, h_bsf, m * sizeof(double), cudaMemcpyHostToDevice);

	// kim and keogh result structures
	struct kim_results * kim_res;
	cudaMallocManaged(&kim_res, (EPOCH) * sizeof(struct kim_results));

	struct keogh_results * keogh_res;
	cudaMallocManaged(&keogh_res, (EPOCH) * sizeof(struct keogh_results));

	// Copy all the Query related arrays
	cudaMemcpy(d_q, h_q, m * sizeof(double), cudaMemcpyHostToDevice);
	bool done = false;
	bool last = false;
	int it = 0, ep = 0, k = 0;
	
	while (!done) {
		/// Read first m-1 points
		if (it == 0) {
			initKernel<<<N, M, m * sizeof(double)>>>(d_q, m);
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
			sh++;
			cudaMemcpy(d_buffer, h_buffer, ep * sizeof(double), cudaMemcpyHostToDevice); // to copy from CPU to GPU
			// Just for printing a dot for approximate a million point. Not much accurate.

			// Do everything here
			// processKernel<<<N, M, m * sizeof(double)>>>(d_q, d_buffer, d_cost,
			// 		d_cost_prev, d_bsf, d_loc, d_t, d_tz, m, r, bsf, ep, EPOCH,
			// 		d_order, d_uo, d_lo, d_cb1);
			// Do the next set of buffering
			// printf("Kernel done.\n");


			epoch = EPOCH;
			ep = 0;
			while (ep < epoch) {
				if (fscanf(fp, "%lf", &d) == EOF) {
					last = true;
					break;
				}
				h_buffer[ep] = d;
				ep++;
			}
			// printf("Loading next set done\n");
			cudaMemcpy(h_bsf, d_bsf, size * sizeof(double),
					cudaMemcpyDeviceToHost);

			cudaMemcpy(h_loc, d_loc, size * sizeof(int),
					cudaMemcpyDeviceToHost);

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
			//printf("Computation Done.\n");


		}
		it++;
	}
	fclose(fp);

	free(h_q);
	free(h_buffer);
	free(h_bsf);

	free(uo);
	free(lo);
	free(order);
	free(u);
	free(l);
	free(cb1);

	cudaFree(d_buffer);
	cudaFree(d_q);
	cudaFree(d_bsf);
	cudaFree(d_loc);
	cudaFree(d_cost);
	cudaFree(d_cost_prev);
	cudaFree(d_t);
	cudaFree(d_tz);

	cudaFree(d_uo);
	cudaFree(d_lo);
	cudaFree(d_order);
	cudaFree(d_cb1);

	printf("\n");

	// Note that loc and i are long long.
	// cout << "Location : " << loc << endl;
	// cout << "Distance : " << sqrt(bsf) << endl;
    // cout << "Data Scanned : " << i << endl;
	// cout << "Total Execution Time : " << (t2-t1)/CLOCKS_PER_SEC << " sec" << endl;

	printf("Distance  %lf\n", sqrt(bsf));
	printf("Location %lld\n", loc);
	//printf("No of iterations %d\n", sh);
	return 0;
}

