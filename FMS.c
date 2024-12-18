#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include "murmurhash.h"
#include "FMS.h"


// Function to set a bit in the bitmap
void setBit(FM_Sketch *fm, uint32_t index, uint32_t position) {
    fm->bitmap[index] |= (1U << position);  // Set the bit at 'position' in 'index' row
}

void generateRandomTuples(Tuple *tuples, size_t numTuples) {
    for (size_t i = 0; i < numTuples; i++) {
        tuples[i].a = rand() % 100;  // Example range
        tuples[i].b = rand() % 100;
        tuples[i].c = rand() % 100;
    }
}

void *encodeTuplesLinkedList(void *arg) {
    ThreadArg *threadArg = (ThreadArg *)arg;
    Node *current = threadArg->startNode; // startNode points to the first node for this thread
    while (current != threadArg->endNode) { // endNode is the stopping point (exclusive)
        Tuple t = current->tuple;
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "%d,%d,%d", t.a, t.b, t.c);
        uint32_t hash = murmurhash3(buffer, strlen(buffer), threadArg->seed);
        uint32_t r = hash % M;
        uint32_t q = hash / M;
        uint32_t position = 0;
        while ((q & 1) == 0 && position < W) {
            q >>= 1;
            position++;
        }
        if (position < W) {
            setBit(threadArg->fm, r, position);
        }
        current = current->next;
    }
    return NULL;
}

void processTuples(FM_Sketch* fm, Node* head, int numThreads) {
    srand((unsigned int)time(NULL));
    uint32_t randomSeed = rand();

    pthread_t* threads = malloc(numThreads * sizeof(pthread_t));
    ThreadArg* threadArgs = malloc(numThreads * sizeof(ThreadArg));

    // Calculate numItems by traversing the list
    size_t numItems = 0;
    Node* temp = head;
    while (temp) {
        numItems++;
        temp = temp->next;
    }

    size_t itemsPerThread = numItems / numThreads;
    Node *current = head;
    size_t count = 0;

    // Dividing linked list for each thread
    for (int i = 0; i < numThreads; i++) {
        threadArgs[i].fm = fm;
        threadArgs[i].startNode = current;
        threadArgs[i].seed = randomSeed;

        // Advance current pointer for the next thread
        size_t limit = (i == numThreads - 1) ? numItems : (i + 1) * itemsPerThread;
        while (current != NULL && count < limit) {
            current = current->next;
            count++;
        }
        threadArgs[i].endNode = current; // The next thread starts from here

        pthread_create(&threads[i], NULL, encodeTuplesLinkedList, (void *)&threadArgs[i]);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(threadArgs);
}

void printFM(const FM_Sketch *fm) {
    printf("FM Sketch Bitmap:\n");
    for (int i = 0; i < M; ++i) {
        uint32_t row = fm->bitmap[i];
        for (int j = 0; j < W; ++j) {
            // Print 1 if the bit is set, 0 otherwise
            printf("%d", (row & (1U << j)) ? 1 : 0);
        }
        printf("\n"); // Newline after each row
    }
}

void initFM(FM_Sketch *fm) {
    memset(fm->bitmap, 0, sizeof(fm->bitmap));
}

double estimateCardinality(const FM_Sketch *fm) {
    double phi = 0.77351;
    double S = 0.0;

    // Summing the position of the first 0 bit for each row in the bitmap
    for (int i = 0; i < M; ++i) {
        uint32_t row = fm->bitmap[i];
        int R = 0;
        while (R < W && (row & (1U << R))) {
            ++R;
        }
        S += R;
    }

    double averageR = S / M;
    // Correctly applying the formula as per the Python implementation
    double Card = (pow(2.0, averageR) - pow(2.0, -1.75 * averageR)) * M / phi;
    return Card;
}

//int main() {
//    #define M 1024
//    #define W 32
//    #define NUM_THREADS 10
//    #define NUM_ITEMS 10000  // Number of tuples
//    srand((unsigned int)time(NULL));
//    uint32_t randomSeed = rand();
//    Tuple tuples[NUM_ITEMS];
//    generateRandomTuples(tuples, NUM_ITEMS);
//
//    FM_Sketch fm = {0};
//    pthread_t threads[NUM_THREADS];
//    ThreadArg threadArgs[NUM_THREADS];
//    size_t itemsPerThread = NUM_ITEMS / NUM_THREADS;
//
//    for (int i = 0; i < NUM_THREADS; i++) {
//        threadArgs[i].fm = &fm;
//        threadArgs[i].tuples = tuples;
//        threadArgs[i].start = i * itemsPerThread;
//        threadArgs[i].end = (i + 1) * itemsPerThread;
//        threadArgs[i].seed = randomSeed; // Example seed
//
//        if (i == NUM_THREADS - 1) {
//            threadArgs[i].end = NUM_ITEMS; // Last thread processes remaining items
//        }
//        pthread_create(&threads[i], NULL, encodeTuples, (void *)&threadArgs[i]);
//    }
//
//    for (int i = 0; i < NUM_THREADS; i++) {
//        pthread_join(threads[i], NULL);
//    }
//
//    printFM(&fm);
//    // Once all tuples have been encoded, estimate the cardinality
//    double estimatedCardinality = estimateCardinality(&fm);
//
//    // Print out the estimated cardinality
//    printf("Estimated Cardinality: %f\n", estimatedCardinality);
//
//    return 0;
//}

