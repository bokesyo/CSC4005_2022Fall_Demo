#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif


#define bound_x 4000
#define bound_y 4000
#define max_mass 10
#define err 1e-5f
#define dt 0.0005f
#define gravity_const 100000000.0f
#define radius2 4.0f


int block_size = 512;


int n_body;
int n_iteration;


__global__ void update_position(float *x, float *y, float *vx, float *vy, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
      /* Fetch data */
      float x_ = x[i];
      float y_ = y[i];
      float vx_ = vx[i];
      float vy_ = vy[i];

      x_ += vx_ * dt;
      y_ += vy_ * dt;

      /* Out of bound handling */
      if (x_ > bound_x) {
          x_ = bound_x;
          vx_ = -vx_;
      } else if (x_ < 0) {
          x_ = 0;
          vx_ = -vx_;
      } else if (y_ > bound_y){
          y_ = bound_y;
          vy_ = -vy_;
      } else if (y_ < 0){
          y_ = 0;
          vy_ = -vy_;
      }

      /* Update */

      x[i] = x_;
      y[i] = y_;
      vx[i] = vx_;
      vy[i] = vy_;
    }
}

__global__ void update_velocity(float *m, float *x, float *y, float *vx, float *vy, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {

    /* Initial external force is 0 */
    float Fx = 0.0f; 
    float Fy = 0.0f;

    /* Fetch the body i's data from global memory directly */
    float m_i = m[i];
    float x_i = x[i];
    float y_i = y[i];
    float vx_i = vx[i];
    float vy_i = vy[i];

    /* For each body */
    for (int j = 0; j < n; j++) {

      float m_j = m[j];
      float x_j = x[j];
      float y_j = y[j];
      float vx_j = vx[j];
      float vy_j = vy[j];

      /* Usual case, calculate gravitivity force*/
      float d_x = x_j - x_i;
      float d_y = y_j - y_i;
      float d2 = d_x * d_x + d_y * d_y + err;
      float d_inv = 1 / sqrt(d2);
      float F = gravity_const * m_i * m_j * d_inv * d_inv * d_inv;
      Fx += F * d_x;
      Fy += F * d_y;

      /* Collision handling */
      if (d2 < radius2){
        if (i != j){
          /* F += 2 * m1 * m2 / (m1 + m2) / dt * (vi - vj) */
          float coef = 2 * m_i * m_j / (m_i + m_j) / dt;
          Fx += coef * (vx_j - vx_i);
          Fy += coef * (vy_j - vy_i);
        }
      }
      
    }

    vx[i] += dt * Fx / m_i;
    vy[i] += dt * Fy / m_i;

  }
}


void generate_data(float *m, float *x,float *y,float *vx,float *vy, int n) {
    // TODO: Generate proper initial position and mass for better visualization
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        m[i] = rand() % max_mass + 1.0f;
        x[i] = 2000.0f + rand() % (bound_x / 4);
        y[i] = 2000.0f + rand() % (bound_y / 4);
        vx[i] = 0.0f;
        vy[i] = 0.0f;
    }
}



void master() {
    float* m = new float[n_body];
    float* x = new float[n_body];
    float* y = new float[n_body];
    float* vx = new float[n_body];
    float* vy = new float[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    // Logger l = Logger("cuda", n_body, bound_x, bound_y);

    float *device_m;
    float *device_x;
    float *device_y;
    float *device_vx;
    float *device_vy;

    cudaMalloc(&device_m, n_body * sizeof(float));
    cudaMalloc(&device_x, n_body * sizeof(float));
    cudaMalloc(&device_y, n_body * sizeof(float));
    cudaMalloc(&device_vx, n_body * sizeof(float));
    cudaMalloc(&device_vy, n_body * sizeof(float));

    cudaMemcpy(device_m, m, n_body * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, x, n_body * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, y, n_body * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vx, vx, n_body * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vy, vy, n_body * sizeof(float), cudaMemcpyHostToDevice);

    int n_block = n_body / block_size + 1;

    for (int i = 0; i < n_iteration; i++){
        // std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        update_velocity<<<n_block, block_size>>>(device_m, device_x, device_y, device_vx, device_vy, n_body);
        update_position<<<n_block, block_size>>>(device_x, device_y, device_vx, device_vy, n_body);

        cudaMemcpy(x, device_x, n_body * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(y, device_y, n_body * sizeof(float), cudaMemcpyDeviceToHost);

        // l.save_frame(x, y);

        // std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<float> time_span = t2 - t1;
        
        // printf("Iteration %d, elapsed time: %.3f\n", i, time_span);

        #ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        float xi;
        float yi;
        for (int i = 0; i < n_body; i++){
            xi = x[i];
            yi = y[i];
            glVertex2f(xi, yi);
        }
        glEnd();
        glFlush();
        glutSwapBuffers();
        #else

        #endif

    }

    cudaFree(device_m);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_vx);
    cudaFree(device_vy);

    delete m;
    delete x;
    delete y;
    delete vx;
    delete vy;
    
}


int main(int argc, char *argv[]){
    
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation CUDA Implementation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, bound_x, 0, bound_y);
    glutDisplayFunc(master);
    glutMainLoop();
    #endif



    printf("Student ID: 119010001\n"); // replace it with your student id
    printf("Name: Your Name\n"); // replace it with your name
    printf("Assignment 2: N Body Simulation CUDA Implementation\n");

    return 0;

}


