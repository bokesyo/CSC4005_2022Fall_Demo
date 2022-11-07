#include <cstdio>
#include <omp.h>

int main() {
    omp_set_num_threads(2);
#pragma omp parallel
    {
        printf("Double!\n");
#pragma omp single
        printf("Single!\n");
        printf("Double!\n");
    }

    return 0;
}
