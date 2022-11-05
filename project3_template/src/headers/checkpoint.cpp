#include <stdio.h>
#include <time.h>

class Logger {
    public:
        int n_body;
        int x_bound;
        int y_bound;
        int current_iteration;
        char* version;
        char* start_time;
        Logger();
        void save_frame(double* x, double* y);
        void update_metadata();
        ~Logger();
};


// not yet implemented


// int main() {
//     Logger a;
    
//     return 0;
// }
