/*
    Here is a tool to help you store running results to file system for further analysis and reproducing.
    API:
        Create a logger: 
            Logger l = Logger(const char* version, int n_body_, int x_bound_, int y_bound_);
        Save a new frame:
            l.save_frame(double* x, double* y);
    Example:
        Logger l = Logger("cuda", 10000, 4000, 4000);
        for (int i = 0; i < n_iterations; i++){
            // compute x,y
            l.save_frame(x, y);
        }
*/

#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>


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
        int is_exist(const char* path);
};


/* Implementation */


int Logger::is_exist(const char* path){
    return !access(path, F_OK);
}


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
    f.close();
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
    f.close();
    return;
};

