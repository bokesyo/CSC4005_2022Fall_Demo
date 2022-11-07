#include <cstdio>
#include <omp.h>

int main() {
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("[%d] Part 1\n", id);

#pragma omp barrier

        printf("[%d] Part 2\n", id);
    }

    return 0;
}
