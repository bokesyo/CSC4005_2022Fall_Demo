#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"


int block_size = 512; // cuda thread block size
int size; // problem size


__global__ void initialize(float *data) {
    // TODO: intialize the temperature distribution (in parallelized way)
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i < n) {
    // }
}


__global__ void generate_fire_area(bool *fire_area){
    // TODO: generate the fire area (in parallelized way)
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i < n) {
    // }
}


__global__ void update(float *data, float *new_data) {
    // TODO: update temperature for each point  (in parallelized way)
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i < n) {
    // }
}


__global__ void maintain_wall(float *data) {
    // TODO: maintain the temperature of the wall (sequential is enough)
}


__global__ void maintain_fire(float *data, bool *fire_area) {
    // TODO: maintain the temperature of the fire (in parallelized way)
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i < n) {  
    // }
}


bool check_continue(float *data, float *new_data) {
    // TODO: determine if terminated (in parallelized way) you may need to define extra __device__ and __global__ functions
    return true;
}


__global__ void data2pixels(float *data, float *pixels){
    // TODO: convert rawdata (large, size^2) to pixels (small, resolution^2) for faster rendering speed (in parallelized way)
    
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


void master() {
    float *data_odd;
    float *data_even;
    bool *fire_area;
    float *pixels;
    float *host_pixels;

    host_pixels = new float[resolution * resolution];

    cudaMalloc(&data_odd, size * size * sizeof(float));
    cudaMalloc(&data_even, size * size * sizeof(float));
    cudaMalloc(&fire_area, size * size * sizeof(bool));
    cudaMalloc(&pixels, resolution * resolution * sizeof(float));

    int n_block_size = size * size / block_size + 1;
    int n_block_resolution = resolution * resolution / block_size + 1;

    initialize<<<n_block_size, block_size>>>(data_odd);
    generate_fire_area<<<n_block_size, block_size>>>(fire_area);
    
    int count = 1;
    bool cont = true; // if should continue
    double total_time = 0;

    while (true){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // TODO: modify the following lines to fit your need.
        if (count % 2 == 1) {
            update<<<n_block_size, block_size>>>(data_odd, data_even);
            maintain_fire<<<n_block_size, block_size>>>(data_even, fire_area);
            maintain_wall<<<1, 1>>>(data_even);
            cont = check_continue(data_odd, data_even);
            data2pixels<<<n_block_resolution, block_size>>>(data_even, pixels);
        } else {
            update<<<n_block_size, block_size>>>(data_even, data_odd);
            maintain_fire<<<n_block_size, block_size>>>(data_odd, fire_area);
            maintain_wall<<<1, 1>>>(data_odd);
            cont = check_continue(data_odd, data_even);
            data2pixels<<<n_block_resolution, block_size>>>(data_odd, pixels);
        }

        cudaMemcpy(host_pixels, pixels, resolution * resolution * sizeof(float), cudaMemcpyDeviceToHost);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        total_time += this_time;
        plot(host_pixels);
        count++;

    }

    printf("Converge after %d iterations, elapsed time: %.6f, average computation time: %.6f\n", count-1, total_time, (double) total_time / (count-1));


    cudaFree(data_odd);
    cudaFree(data_even);
    cudaFree(pixels);
    cudaFree(fire_area);

    delete[] host_pixels;
    
}


int main(int argc, char *argv[]){
    
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
    printf("Assignment 4: Heat Distribution CUDA Implementation\n");

    return 0;

}


