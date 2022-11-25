#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"


int size; // problem size


int my_rank;
int world_size;


void initialize(float *data) {
    // intialize the temperature distribution
    int len = size * size;
    for (int i = 0; i < len; i++) {
        data[i] = wall_temp;
    }
}


void generate_fire_area(bool *fire_area){
    // generate the fire area
    int len = size * size;
    for (int i = 0; i < len; i++) {
        fire_area[i] = 0;
    }

    float fire1_r2 = fire_size * fire_size;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            int a = i - size / 2;
            int b = j - size / 2;
            int r2 = 0.5 * a * a + 0.8 * b * b - 0.5 * a * b;
            if (r2 < fire1_r2) fire_area[i * size + j] = 1;
        }
    }

    float fire2_r2 = (fire_size / 2) * (fire_size / 2);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            int a = i - 1 * size / 3;
            int b = j - 1 * size / 3;
            int r2 = a * a + b * b;
            if (r2 < fire2_r2) fire_area[i * size + j] = 1;
        }
    }
}


void update(float *data, float *new_data, int begin, int end) {
    // TODO: update the temperature of each point, and store the result in `new_data` to avoid data racing
    
}


void maintain_fire(float *data, bool* fire_area, int begin, int end) {
    // TODO: maintain the temperature of fire
}


void maintain_wall(float *data, int begin, int end) {
    // TODO: maintain the temperature of the wall
}


bool check_continue(float *data, float *new_data){
    // TODO: determine if we should stop (because the temperature distribution will finally converge)
    return true;
}


void data2pixels(float *data, float *pixels){
    // convert rawdata (large, size^2) to pixels (small, resolution^2) for faster rendering speed
    float factor = (float) size / resolution;
    for (int x = 0; x < resolution; x++){
        for (int y = 0; y < resolution; y++){
            int x_raw = (int) (x * factor);
            int y_raw = (int) (y * factor);
            int idx_rawdata = y_raw * size + x_raw;
            float temp = data[idx_rawdata];
            float color = (float) ((int) temp / 5 * 5) / fire_temp;
            pixels[x * resolution + y] = color;
        }
    }
}


void plot(float* pixels){
    // visualize temprature distribution
    #ifdef GUI
    glClear(GL_COLOR_BUFFER_BIT);
    float particle_size = (float) window_size / resolution;
    glPointSize(particle_size);
    glBegin(GL_POINTS);
    for (int x = 0; x < resolution; x++){
        for (int y = 0; y < resolution; y++){
            float color = pixels[x * resolution + y];
            glColor3f(color, 1.0f - color, 1.0f - color);
            glVertex2f(x, y);
        }
    }
    glEnd();
    glFlush();
    glutSwapBuffers();
    #endif
}



void slave(){
    // TODO: MPI routine (one possible solution, you can use another partition method)
    int my_begin_row_id = size * my_rank / (world_size);
    int my_end_row_id = size * (my_rank + 1) / world_size;
    float* local_data;
    float* pixcels;

    while (true) {

    }
    // TODO End
}



void master() {
    // TODO: MPI routine (one possible solution, you can use another partition method)
    float* data_odd = new float[size * size];
    float* data_even = new float[size * size];
    float* pixels = new float[resolution * resolution];
    bool* fire_area = new bool[size * size];

    initialize(data_odd);
    generate_fire_area(fire_area);

    bool cont = true;
    int count = 1;
    double total_time = 0;

    while (true) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        // MPI Routine
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        total_time += this_time;
        printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        count++;

        #ifdef GUI
        // plot(pixels);
        #endif
    }

    delete[] data_odd;
    delete[] data_even;
    delete[] fire_area;
    delete[] pixels;
}




int main(int argc, char *argv[]) {
    size = atoi(argv[1]);

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


	if (my_rank == 0) {
        #ifdef GUI
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(window_size, window_size);
        glutCreateWindow("Heat Distribution Simulation Sequential Implementation");
        gluOrtho2D(0, resolution, 0, resolution);
        #endif
        master();
	} else {
        slave();
    }

	if (my_rank == 0){
		printf("Student ID: 119010001\n"); // replace it with your student id
		printf("Name: Your Name\n"); // replace it with your name
		printf("Assignment 3: Heat Distribution Simulation MPI Implementation\n");
	}

	MPI_Finalize();

	return 0;
}

