#include <cstdlib>
#include <fstream>
#include <iostream>

int main (int argc, char **argv){
    int num_elements; // number of elements to be sorted
    num_elements = atoi(argv[1]); // convert command line argument to num_elements

    int elements[num_elements]; // store elements
    std::ifstream input(argv[2]);
        int element;
        int i = 0;
        while (input >> element) {
            elements[i] = element;
            i++;
        }
    
    int unsort_count = 0;
    for (int i = 0; i < num_elements-1; i++) {
        if (elements[i] > elements[i + 1])
            unsort_count++;
    }

    if (unsort_count < 1) {
        std::cout << "Sorted." << std::endl;
    } else {
        std::cout << "Not Sorted. " << unsort_count << " errors." << std::endl;
    }

}