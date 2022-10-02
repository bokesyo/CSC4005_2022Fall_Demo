#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>


int main (int argc, char **argv){

    MPI_Init(&argc, &argv); 

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_elements; // number of elements to be sorted
    
    num_elements = atoi(argv[1]); // convert command line argument to num_elements

    int elements[num_elements]; // store elements
    int sorted_elements[num_elements]; // store sorted elements

    if (rank == 0) { // read inputs from file (master process)
        std::ifstream input(argv[2]);
        int element;
        int i = 0;
        while (input >> element) {
            elements[i] = element;
            i++;
        }
        std::cout << "actual number of elements:" << i << std::endl;
    }

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    if (rank == 0){ 
        t1 = std::chrono::high_resolution_clock::now(); // record time
    }

    /* TODO BEGIN
        Implement parallel odd even transposition sort
        Code in this block is not a necessary. 
        Replace it with your own code.
        Useful MPI documentation: https://rookiehpc.github.io/mpi/docs
    */

    int num_my_element = num_elements / world_size; // number of elements allocated to each process
    int my_element[num_my_element]; // store elements of each process
    MPI_Scatter(elements, num_my_element, MPI_INT, my_element, num_my_element, MPI_INT, 0, MPI_COMM_WORLD); // distribute elements to each process
    MPI_Gather(my_element, num_my_element, MPI_INT, sorted_elements, num_my_element, MPI_INT, 0, MPI_COMM_WORLD); // collect result from each process
    
    /* TODO END */

    if (rank == 0){ // record time (only executed in master process)
        t2 = std::chrono::high_resolution_clock::now();  
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Student ID: " << "119010001" << std::endl; // replace it with your student id
        std::cout << "Name: " << "Your Name" << std::endl; // replace it with your name
        std::cout << "Assignment 1" << std::endl;
        std::cout << "Run Time: " << time_span.count() << " seconds" << std::endl;
        std::cout << "Input Size: " << num_elements << std::endl;
        std::cout << "Process Number: " << world_size << std::endl; 
    }

    if (rank == 0){ // write result to file (only executed in master process)
        std::ofstream output(argv[2]+std::string(".parallel.out"), std::ios_base::out);
        for (int i = 0; i < num_elements; i++) {
            output << sorted_elements[i] << std::endl;
        }
    }
    
    MPI_Finalize();
    
    return 0;
}


