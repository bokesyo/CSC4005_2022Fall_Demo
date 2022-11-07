#include <iostream>
#include <omp.h>
#include <vector>

int main() {
#pragma omp parallel
    { std::cout << "Thread number: " << omp_get_thread_num() << std::endl; }
    return 0;
}
