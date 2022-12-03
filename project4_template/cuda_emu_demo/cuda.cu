#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include <chrono>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#define threshold 0.00001f
#define fire_temp 90.0f
#define wall_temp 0.0f
#define fire_size 20


int block_size = 512; // cuda thread block size
int size; // problem size


__global__ void initialize(float *data, int size) {
    // TODO: intialize the temperature distribution (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (size*size)) {
        data[i] = wall_temp;
    }
}


__global__ void generate_fire_area(bool *fire_area, int size){
    // TODO: generate the fire area (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (size*size)) {
        fire_area[i] = 0;
        float fire1_r2 = fire_size * fire_size;
        int x = i / size;
        int y = i % size;
        int a = x - size / 2;
        int b = y - size / 2;
        int r2 = 0.5 * a * a + 0.8 * b * b - 0.5 * a * b;
        if (r2 < fire1_r2) fire_area[i] = 1;
    }
}


__global__ void update(float *data, float *new_data, int size) {
    // TODO: update temperature for each point  (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (size*size)) {
        if ((i > size) && (i < (size*size-size-1))){
            float up = data[i - size];
            float down = data[i + size];
            float left = data[i - 1];
            float right = data[i + 1];
            float new_val = (up + down + left + right) / 4;
            new_data[i] = new_val;
        }
        
    }
}


__global__ void maintain_wall(float *data, int size) {
    // TODO: maintain the temperature of the wall (sequential is enough)
    
    
}


__global__ void maintain_fire(float *data, bool *fire_area, int size) {
    // TODO: maintain the temperature of the fire (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (size*size)) {
        if (fire_area[i]) data[i] = fire_temp;
    }
}


#ifdef GUI
__global__ void data2pixels(float *data, GLubyte* pixels, int size){
    // TODO: convert rawdata (large, size^2) to pixels (small, size^2) for faster rendering speed (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (size*size)) {
        float temp = data[i];
        int color =  ((int) temp / 5 * 5) * (float) 255 / fire_temp;
        pixels[i * 3] = color;
        pixels[i * 3 + 1] = 255 - color;
        pixels[i * 3 + 2] = 255 - color;
    }
    
}


void plot(GLubyte* pixels){
    // visualize temprature distribution
    #ifdef GUI
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(size, size, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    glutSwapBuffers();
    #endif
}
#endif


void master() {
    float *data_odd;
    float *data_even;
    bool *fire_area;

    cudaMalloc(&data_odd, size * size * sizeof(float));
    cudaMalloc(&data_even, size * size * sizeof(float));
    cudaMalloc(&fire_area, size * size * sizeof(bool));

    #ifdef GUI
    GLubyte *pixels;
    GLubyte *host_pixels;
    host_pixels = new GLubyte[size * size * 3];
    cudaMalloc(&pixels, size * size * 3 * sizeof(GLubyte));
    #endif

    int n_block_size = size * size / block_size + 1;

    initialize<<<n_block_size, block_size>>>(data_odd, size);
    generate_fire_area<<<n_block_size, block_size>>>(fire_area, size);
    
    int count = 1;
    double total_time = 0;

    while (true){
        // std::chrono::high_size_clock::time_point t1 = std::chrono::high_size_clock::now();

        // TODO: modify the following lines to fit your need.
        if (count % 2 == 1) {
            update<<<n_block_size, block_size>>>(data_odd, data_even, size);
            maintain_fire<<<n_block_size, block_size>>>(data_even, fire_area, size);
            maintain_wall<<<1, 1>>>(data_even, size);
        } else {
            update<<<n_block_size, block_size>>>(data_even, data_odd, size);
            maintain_fire<<<n_block_size, block_size>>>(data_odd, fire_area, size);
            maintain_wall<<<1, 1>>>(data_odd, size);
        }

        // std::chrono::high_size_clock::time_point t2 = std::chrono::high_size_clock::now();
        // double this_time = std::chrono::duration<double>(t2 - t1).count();
        // total_time += this_time;
        // printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        count++;
        
        #ifdef GUI
        if (count % 2 == 1) {
            data2pixels<<<n_block_size, block_size>>>(data_even, pixels, size);
        } else {
            data2pixels<<<n_block_size, block_size>>>(data_odd, pixels, size);
        }
        cudaMemcpy(host_pixels, pixels, size * size * 3 * sizeof(GLubyte), cudaMemcpyDeviceToHost);
        plot(host_pixels);
        #endif

    }

    printf("Converge after %d iterations, elapsed time: %.6f, average computation time: %.6f\n", count-1, total_time, (double) total_time / (count-1));


    cudaFree(data_odd);
    cudaFree(data_even);
    cudaFree(fire_area);

    #ifdef GUI
    cudaFree(pixels);
    delete[] host_pixels;
    #endif
    
}


int main(int argc, char *argv[]){
    
    size = atoi(argv[1]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(size, size);
    glutCreateWindow("Heat Distribution Simulation Sequential Implementation");
    gluOrtho2D(0, size, 0, size);
    #endif

    master();

    printf("Student ID: 119010001\n"); // replace it with your student id
    printf("Name: Your Name\n"); // replace it with your name
    printf("Assignment 4: Heat Distribution CUDA Implementation\n");

    return 0;

}



