/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long int d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram_cpu;		/* list of all buckets in the histogram   */
bucket * histogram_gpu;	
bucket * histogram_temp;	
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list_cpu;		/* list of all data points                */
atom * atom_list_gpu;

int blockSize;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list_cpu[ind1].x_pos;
	double x2 = atom_list_cpu[ind2].x_pos;
	double y1 = atom_list_cpu[ind1].y_pos;
	double y2 = atom_list_cpu[ind2].y_pos;
	double z1 = atom_list_cpu[ind1].z_pos;
	double z2 = atom_list_cpu[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


__device__ double p2p_distance_gpu(atom &currentThreadVal, atom &R) {
	
	double x1 = currentThreadVal.x_pos;
	double x2 = R.x_pos;
	double y1 = currentThreadVal.y_pos;
	double y2 = R.y_pos;
	double z1 = currentThreadVal.z_pos;
	double z2 = R.z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline_cpu() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram_cpu[h_pos].d_cnt++;
		} 
	}
	return 0;
}

__global__ void PDH_baseline_gpu(long long pdh_ac, double res, bucket *histogram_gpu, atom *atom_list_gpu, int num_buckets){
	int i, j, h_pos;
		
	double dist;
	extern __shared__ atom R[];

	int blkDim = blockDim.x;
	int thdIdx = threadIdx.x;
	int blkIdx = blockIdx.x;
	atom currentThreadVal;


	bucket* sharedHistogramGPU = (bucket *)&R[blkDim];
	
	long long tid = blockDim.x * blockIdx.x + threadIdx.x;
	

	for(i = thdIdx; i <  num_buckets; i = i + blkDim)
	{
		sharedHistogramGPU[i].d_cnt = 0;
	}
	__syncthreads();
	currentThreadVal = atom_list_gpu[tid];
	

	for(i = blkIdx + 1 ; i < gridDim.x ; i++) 
	{
		int blockStartIdx = i * blkDim;
		R[thdIdx] = atom_list_gpu[thdIdx + blockStartIdx];
		__syncthreads();

		if(blockStartIdx < pdh_ac)
		{
			for(j = 0; j < blkDim; j++) 
			{
				if(j + blockStartIdx < pdh_ac)
				{
					dist = p2p_distance_gpu(currentThreadVal, R[j]);
					h_pos = (int) (dist / res);
					atomicAdd(&(sharedHistogramGPU[h_pos].d_cnt), 1);	 				
				}
			}
			
		}
		__syncthreads();

	}


	R[thdIdx] = currentThreadVal;
	__syncthreads();

	int halfBlkDim = (int)(blkDim/2);
	halfBlkDim = (blkDim%2==0 && thdIdx >= halfBlkDim)? halfBlkDim - 1 : halfBlkDim;

	if(tid < pdh_ac)
	{
		for(j = 0; j < halfBlkDim; j++)
		{
			int index = (thdIdx + j + 1) % blkDim;
			if(index + ( blkDim * blkIdx ) < pdh_ac)
			{
				dist = p2p_distance_gpu(currentThreadVal, R[index]);
				h_pos = (int) (dist / res);
				atomicAdd(&(sharedHistogramGPU[h_pos].d_cnt), 1);
			}
		}
		
	}
	__syncthreads();
	for(i = thdIdx; i <  num_buckets; i = i + blkDim)
	{
		atomicAdd(&(histogram_gpu[i].d_cnt), (sharedHistogramGPU[i].d_cnt));
	}
}



/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket *hist){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", hist[i].d_cnt);
		total_cnt += hist[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void compute_print_hist_diff(bucket *cpu, bucket *gpu){
	int i; 
	long long int diff = 0; 	
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		diff = abs((long long int)(cpu[i].d_cnt - gpu[i].d_cnt));
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", diff);
		total_cnt += diff;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	blockSize = atof(argv[3]);
	//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

	histogram_cpu = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list_cpu = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list_cpu[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list_cpu[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list_cpu[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	/* -------------------------------- CPU Execution Starts ------------------------------ */
	/* start counting time */
	gettimeofday(&startTime, &Idunno);

	/* call CPU single thread version to compute the histogram */
	PDH_baseline_cpu();

	/* check the total running time */ 
	report_running_time();

	/* print out the histogram */
	output_histogram(histogram_cpu);

	/* -------------------------------- CPU Execution Ends -------------------------------- */

	/* -------------------------------- GPU Execution Starts ------------------------------ */
	

	histogram_gpu = (bucket *)malloc(sizeof(bucket)*num_buckets);
	histogram_temp = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list_gpu = (atom *)malloc(sizeof(atom)*PDH_acnt);

	cudaMalloc((void**)&atom_list_gpu, sizeof(atom) * PDH_acnt);
	cudaMemcpy(atom_list_gpu, atom_list_cpu, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&histogram_gpu, sizeof(bucket) * num_buckets);
	cudaMemcpy(histogram_gpu, histogram_temp,sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int no_of_blocks = ceil(PDH_acnt/(float)blockSize);
	
	
	PDH_baseline_gpu<<< no_of_blocks , blockSize, sizeof(bucket) * num_buckets + sizeof(atom) * blockSize>>>(PDH_acnt, PDH_res, histogram_gpu, atom_list_gpu, num_buckets);


	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	float elapsedTime; 
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf( "******** Total Running Time of Kernel: %0.5f ms *******\n", elapsedTime );
	cudaEventDestroy(start); 
	cudaEventDestroy(stop); 
	
	cudaMemcpy(histogram_temp, histogram_gpu, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);	
	/* print out the histogram */
	output_histogram(histogram_temp);

	/* print out the difference between cpu and gpu results */
	printf("\nDifference between CPU and GPU execution results - \n");
	compute_print_hist_diff(histogram_cpu, histogram_temp);

	/* print out the histogram ends*/

	cudaFree(histogram_gpu);
	cudaFree(histogram_temp);
	cudaFree(atom_list_gpu);

	free(histogram_cpu);

	/* -------------------------------- GPU Execution Ends -------------------------------- */
	
	return 0;
}