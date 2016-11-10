#include "circularQueue.h"

void init(buffer_array_t buf) {
	sem_init(&buf.countsem, 0, 0);
	sem_init(&buf.spacesem, 0, N);
	sem_init(&buf.lock, 0, 1);
	buf.in = 0;
	buf.out = 0;
}

void enqueue(buffer_array_t buf, buffer_candidate_t cand){
	sem_wait(&buf.spacesem);
		sem_wait(&buf.lock);
			buf.queue[(buf.in++) & (N-1)] = cand;
		sem_post(&buf.lock);
	sem_post(&buf.countsem);
}

buffer_candidate_t dequeue(buffer_array_t buf){
	sem_wait(&buf.countsem);
		sem_wait(&buf.lock);
			buffer_candidate_t result = buf.queue[(buf.out++) & (N-1)];
		sem_post(&buf.lock);
	sem_post(&buf.spacesem);
	return result;
}
