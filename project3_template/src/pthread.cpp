#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "physics.h"


int n_thd; // number of threads
int n_body;
int n_iteration;


double* m;
double* x;
double* y;
double* vx;
double* vy;


void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n) {
    // TODO: Generate proper initial position and mass for better visualization
    for (int i = 0; i < n; i++) {
        m[i] = rand() % max_mass + 1.0f;
        x[i] = rand() % bound_x;
        y[i] = rand() % bound_y;
        vx[i] = 0.0f;
        vy[i] = 0.0f;
    }
}


void update_position(double *x, double *y, double *vx, double *vy, int n) {
    //TODO: update position 

}

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int n) {
    //TODO: calculate force and acceleration, update velocity

}


typedef struct {
    //TODO: specify your arguments for threads
    //int a;
    //int b;
    //TODO END
} Args;


void* worker(void* args) {
    //TODO: procedure in each threads
    // Args* my_arg = (Args*) args;
    // int a = my_arg->a;
    // int b = my_arg->b;
    // TODO: pthread routine
    update_velocity(m, x, y, vx, vy, n_body);
    update_position(x, y, vx, vy, n_body);
    // TODO End
    //TODO END
}


void master(){
    m = new double[n_body];
    x = new double[n_body];
    y = new double[n_body];
    vx = new double[n_body];
    vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    for (int i = 0; i < n_iteration; i++){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        //TODO: assign jobs
        //TODO End
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = t2 - t1;

        printf("Iteration %d, elapsed time: %.3f\n", i, time_span);

        #ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        double xi;
        double yi;
        for (int i = 0; i < n_body; i++){
            xi = total_x[i];
            yi = total_y[i];
            glVertex2f(xi, yi);
        }
        glEnd();
        glFlush();
        glutSwapBuffers();
        #else

        #endif
    }


}


int main(int argc, char *argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_thd = atoi(argv[3]);

    #ifdef GUI
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Pthread");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, X_RESN, 0, Y_RESN);
    #endif
    master();

	return 0;
}

