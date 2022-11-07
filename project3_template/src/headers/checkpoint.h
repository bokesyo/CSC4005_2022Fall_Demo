#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

int is_exist(const char* path){
    return !access(path, F_OK);
}

class Logger {
    public:
        int n_body;
        int x_bound;
        int y_bound;
        int current_iteration;
        std::string version;
        std::string start_time;
        std::string root_path = "./checkpoints/";
        std::string path;

        Logger(const char* version, int n_body_, int x_bound_, int y_bound_);

        void save_frame(double* x, double* y);
        void update_metadata();
};

Logger::Logger(const char* version_, int n_body_, int x_bound_, int y_bound_){
    
    version = version_;
    n_body = n_body_;
    x_bound = x_bound_;
    y_bound = y_bound_;
    current_iteration = 0;

    time_t t = time(0); 
    char tmp[32];
    strftime(tmp, sizeof(tmp), "%Y%m%d%H%M%S",localtime(&t)); 
    start_time = tmp;

    path = root_path + version + "_" + std::to_string(n_body) + "_" + start_time + "/";
    std::cout << path << std::endl;

    if (is_exist(root_path.c_str()) == 0) {
        mkdir(root_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    }

    if (is_exist(path.c_str()) == 0) {
        int isCreate = mkdir(path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        if (!isCreate){
            std::cout << "directory creation success" << std::endl;
        } else {
            std::cout << "directory creation failed" << std::endl;
        }
    }

};



void Logger::save_frame(double* x, double* y){
    std::setprecision(15);
    std::string f_path = path + "data.txt";
    std::cout << path << std::endl;
    std::ofstream f(f_path, std::ios::app | std::ios::binary);
    for (int i = 0; i < n_body; i++){
        f << std::to_string(x[i]) << std::endl << std::to_string(y[i]) << std::endl;
    }
    update_metadata();
    return;
};


void Logger::update_metadata(){
    std::string f_path = path + "metadata.txt";
    std::ofstream f(f_path, std::ios::out);
    current_iteration ++;
    f << version << std::endl << n_body << std::endl 
    << x_bound << std::endl << y_bound << std::endl 
    << current_iteration;
    return;
};
