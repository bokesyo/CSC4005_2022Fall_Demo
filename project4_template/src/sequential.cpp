#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <unistd.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif


#include "./headers/physics.h"


int size; // problem size


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


void update(float *data, float *new_data) {
    // update the temperature of each point, and store the result in `new_data` to avoid data racing
    for (int i = 1; i < (size - 1); i++){
        for (int j = 1; j < (size - 1); j++){
            int idx = i * size + j;
            float up = data[idx - size];
            float down = data[idx + size];
            float left = data[idx - 1];
            float right = data[idx + 1];
            float new_val = (up + down + left + right) / 4;
            new_data[idx] = new_val;
        }
    }
}


void maintain_fire(float *data, bool* fire_area) {
    // maintain the temperature of fire
    int len = size * size;
    for (int i = 0; i < len; i++){
        if (fire_area[i]) data[i] = fire_temp;
    }
}


void maintain_wall(float *data) {
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


void master(){

    float *data_odd;
    float *data_even;
    float *pixels;
    bool *fire_area;
    
    data_odd = new float[size * size];
    data_even = new float[size * size];
    pixels = new float[resolution * resolution];
    fire_area = new bool[size * size];

    generate_fire_area(fire_area);
    initialize(data_odd);

    bool cont = true;
    int count = 1;
    double total_time = 0;

    while (cont) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        if (count % 2 == 1) {
            update(data_odd, data_even);
            maintain_fire(data_even, fire_area);
            maintain_wall(data_even);
            cont = check_continue(data_odd, data_even);
        } else {
            update(data_even, data_odd);
            maintain_fire(data_odd, fire_area);
            maintain_wall(data_odd);
            cont = check_continue(data_odd, data_even);
        }
        
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        total_time += this_time;
        printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        count++;

        #ifdef GUI
        if (count % 2 == 1) {
            data2pixels(data_even, pixels);
        } else {
            data2pixels(data_odd, pixels);
        }
        plot(pixels);
        #endif
        
    }

    printf("Converge after %d iterations, elapsed time: %.6f, average computation time: %.6f\n", count-1, total_time, (double) total_time / (count-1));

    delete[] data_odd;
    delete[] data_even;
    delete[] pixels;
    delete[] fire_area;
  
}


int main(int argc, char* argv[]) {
    size = atoi(argv[1]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_size, window_size);
    glutCreateWindow("Heat Distribution Simulation Sequential Implementation");
    gluOrtho2D(0, resolution, 0, resolution);
    #endif

    master();

    printf("Student ID: 119010001\n"); // replace it with your student id
    printf("Name: Your Name\n"); // replace it with your name
    printf("Assignment 4: Heat Distribution Sequential Implementation\n");

    return 0;
}
