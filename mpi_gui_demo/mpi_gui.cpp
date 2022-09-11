/* Author: bokaixu@link.cuhk.edu.cn */

#include <mpi.h>
#include <unistd.h>

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define max_temp 100.0f
#define canvas_size 800
#define particle_size 5.0f

int size;
int rank;
int world_size;


void initialize(float *data, int size) {
  int arr_len = size * size;

  for (int i = 0; i < arr_len; i++) {
    data[i] = rand() % 100;
  }

}


void update(float *data, float *new_data, int size, int lower, int upper) {
  /* Update the temp of each position */

  for (int i = lower ; i < upper; i++) {
      new_data[i] = rand() % 100;
  }

}



void plot(float* data, int size){

  glClear(GL_COLOR_BUFFER_BIT);
  glPointSize(particle_size);
  glBegin(GL_POINTS);

  int arr_len = size * size;

  for (int i = 0; i < arr_len; i++){

      /* Make contours */
      float color = (float) ((int) data[i] / 5 * 5) / max_temp;
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


void slave(){
    /* Basic Info */
    int data_size = size * size;
    int local_n = (data_size + world_size - 1) / world_size;
    int lower = rank * local_n;
    int upper = (rank + 1) * local_n;

    /* Initialize Storage */
    float *data = new float[data_size];
    float *new_data = new float[data_size];
    float *local_data = new float[local_n];

    while (true) {
        /* Broadcast data to all process */
        MPI_Bcast(data, data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        /* Computation Part */
        update(data, new_data, size, lower, upper);

        /* Construct partial data for Gather() */
        int cursor = 0;
        for (int i = lower; i < upper; i++){
            local_data[cursor] = new_data[i];
            cursor++;
        }

        /* Gather data */
        MPI_Gather(local_data, local_n, MPI_FLOAT, data, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    }

    delete[] data;
    delete[] new_data;

    MPI_Finalize();
}


void master(){
    /* Basic Info */
    int data_size = size * size;
    int local_n = (data_size + world_size - 1) / world_size;
    int lower = rank * local_n;
    int upper = (rank + 1) * local_n;

    /* Initialize Storage */
    float *data = new float[data_size];
    float *new_data = new float[data_size];
    float *local_data = new float[local_n];
    initialize(data, size);

    int count = 1;
    double total_time = 0;

    while (true) {
        /* Timing start*/
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        /* Broadcast data to all process */
        MPI_Bcast(data, data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        /* Computation Part */
        update(data, new_data, size, lower, upper);

        /* Construct partial data for Gather() */
        int cursor = 0;
        for (int i = lower; i < upper; i++){
            local_data[cursor] = new_data[i];
            cursor++;
        }

        /* Gather data */
        MPI_Gather(local_data, local_n, MPI_FLOAT, data, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);


        /* Timing end */
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        total_time += this_time;

        /* Plot */
        plot(data, size);
        count ++;

        sleep(1);
    }

    delete[] data;
    delete[] new_data;

    MPI_Finalize();

}


int main(int argc, char* argv[]) {

    /* MPI preparation */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Prepare basic information */
    size = 100;

    /* Main Part */
    if (rank == 0){
        /* GUI initialize */
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(canvas_size, canvas_size);
        glutCreateWindow("MPI + OpenGL");
        glutDisplayFunc(&master);
        glutMainLoop();
    } else {
        slave();
    }

    return 0;

}
