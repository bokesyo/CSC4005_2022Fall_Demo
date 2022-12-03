#include <cuda.h>
#include <cuda_runtime.h> 

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include <chrono>

#define block_size 512
#define threshold 0.01f
#define fire_temp 100.0f
#define wall_temp 20.0f
#define fire_size 0.1f
#define canvas_size 200

int size;


void initialize(float *data, int size) {
  int arr_len = size * size;

  /* Process usual case */
  for (int i = 0; i < arr_len; i++) {
    data[i] = wall_temp;
  }

  /* Process bounds */
  for (int i = 0; i < size; i++){
    data[i] = wall_temp;
    data[i * size] = wall_temp;
    data[arr_len - i * size - 1] = wall_temp;
    data[arr_len - i - 1] = wall_temp;
  }

  /* Process center */
  for (int i = (0.5f - fire_size) * size; i < (0.5f + fire_size) * size; i++){
    for (int j = (0.5f - fire_size) * size; j < (0.5f + fire_size) * size; j++){
      data[size * i + j] = fire_temp;
    }
  }
}


__global__ void update(float *data, float *new_data, int size) {
  /* Update the temp of each position */
  int arr_len = size * size;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if ((i > size) && (i < (arr_len - size - 1))) {
    /* Fetch data */
    float up = data[i - size];
    float down = data[i + size];
    float left = data[i - 1];
    float right = data[i + 1];
    float self = data[i];

    /* Computation */
    float new_val = (up + down + left + right) / 4;

    /* Write new data*/
    new_data[i] = new_val;

  }
}



void deep_copy(float *data, float *new_data, int size, int lower, int upper) {
  /* Copy new_data to data */
  int arr_len = size * size;
  for (int i = lower ; i < upper; i++) {
    /* Transfer data*/
    data[i] = new_data[i];
  }
}


void normalize(float *data, int size) {
  /* Make four bounds and fire place constant temporature */

  int arr_len = size * size;
  /* Process bounds */
  for (int i = 0; i < size; i++){
    data[i] = wall_temp;
    data[i * size] = wall_temp;
    data[arr_len - i * size - 1] = wall_temp;
    data[arr_len - i - 1] = wall_temp;
  }
  /* Process center */
  for (int i = (0.5f - fire_size) * size; i < (0.5f + fire_size) * size; i++){
    for (int j = (0.5f - fire_size) * size; j < (0.5f + fire_size) * size; j++){
      data[size * i + j] = fire_temp;
    }
  }
}


void plot(float* data, int size){
  /* Plot in OpenGL */
  /* Graphic area */
  glClear(GL_COLOR_BUFFER_BIT);
  
  glPointSize((float) canvas_size / size);
  glBegin(GL_POINTS);
  int arr_len = size * size;
  for (int i = 0; i < arr_len; i++){

      /* Make contours */
      float color = (float) ((int) data[i] / 5 * 5) / fire_temp;
      glColor3f(color,1.0f - color,1.0f - color);

      /* Determine position, in % format */ 
      float y = (float) (i / size) / size - 0.5f;
      float x = (float) (i % size) / size - 0.5f;

      /* Plot */
      glVertex2f(x, y);
  }
  glEnd();
  glFlush();
  glutSwapBuffers();
}


void simulation() {

  /* Prepare basic information */
  int data_size = size * size * sizeof(float);
  int n_block = (size * size + block_size - 1) / block_size;

  /* Continuous condition */
  int cont = 1;

  /* Data Storage */
  float *data = new float[size * size];
  float *new_data = new float[size * size];
  float *device_data;
  float *device_new_data;
  cudaMalloc(&device_data, data_size);
  cudaMalloc(&device_new_data, data_size);

  /* Initialize Data */
  initialize(data, size);

  /* Statistics */
  int count = 1;
  double total_time = 0;
  
  while (cont == 1){

    cudaMemcpy(device_data, data, data_size, cudaMemcpyHostToDevice);
    update<<<n_block, block_size>>>(device_data, device_new_data, size);
    cudaMemcpy(new_data, device_new_data, data_size, cudaMemcpyDeviceToHost);
    normalize(new_data, size);
    deep_copy(data, new_data, size, 0, size * size);

    /* Plot */
    plot(data, size);
    count ++;
    // Sleep(10);

  }

  /* Free */
  cudaFree(device_data);
  cudaFree(device_new_data);
  delete[] data;
  delete[] new_data;

}

int main(int argc, char *argv[]){
    /* Prepare basic information */
    size = atoi(argv[1]);

    /* GUI initialize */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(canvas_size, canvas_size);
    glutCreateWindow("Heat Distribution Simulation CUDA + OpenGL Implementation");
    gluOrtho2D(-0.5, 0.5, -0.5, 0.5);
    glutDisplayFunc(&simulation);
    glutMainLoop();

    return 0;
}
