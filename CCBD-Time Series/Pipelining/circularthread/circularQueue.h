#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>
#define N 2

typedef struct{
	long long location;
	float data[128];
} buffer_candidate_t;

typedef struct{
	buffer_candidate_t queue[N];
	sem_t countsem, spacesem, lock;
	int in;
	int out;
} buffer_array_t;

void buf_init(buffer_array_t);
void buf_enqueue(buffer_array_t, buffer_candidate_t);
buffer_candidate_t buf_dequeue(buffer_array_t);
