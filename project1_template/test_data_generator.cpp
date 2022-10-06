#include <cstdlib>
#include <iostream>
#include <ctime>
#include <fstream>

#define random(a, b) (rand() % (b - a) + a)


int main(int argc, char **argv){
    int num_elements; // number of elements to generate
    num_elements = atoi(argv[1]);

    std::ofstream out;
    out.open(argv[2],std::ios_base::out);

    srand((int)time(0));
    for (int i = 0; i < num_elements; i++){
        out << random(1, 99999999) << std::endl;
    }
    out.close();

    return 0;
}

