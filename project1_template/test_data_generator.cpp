#include <cstdlib>
#include <iostream>
#include <ctime>
#include <fstream>

#define random(a, b) (rand() % (b - a) + a)

using namespace std;

int main(int argc, char **argv){
    int num_elements; // number of elements to generate
    num_elements = atoi(argv[1]);

    ofstream out;
    out.open(argv[2],ios::app);

    srand((int)time(0));
    for (int i = 0; i < num_elements; i++)
    {
        out << random(1, 99999999) << endl;
    }
    out.close();
    return 0;
}

