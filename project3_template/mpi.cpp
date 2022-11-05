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


#define gravity_const 1.0f
#define dt 0.0001f
#define error 1e-9f
#define radius2 4.0f
#define bound_x 4000
#define bound_y 4000
#define max_mass 40000000
#define block_size 1024


int n_body;
int n_iteration;


int rank;
int world_size;


double* total_m;
double* total_x;
double* total_y;
double* total_vx;
double* total_vy;


double* local_m;
double* local_x;
double* local_y;
double* local_vx;
double* local_vy;


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


void slave(){
    // TODO: MPI routine
    //MPI_Scatter
    //update_velocity(local_m, local_x, local_y, local_vx, local_vy, n_body);
    //update_position(local_x, local_y, local_vx, local_vy, n_body);
    //MPI_Gather
    // TODO End
}



void master() {
    total_m = new double[n_body];
    total_x = new double[n_body];
    total_y = new double[n_body];
    total_vx = new double[n_body];
    total_vy = new double[n_body];

    generate_data(total_m, total_x, total_y, total_vx, total_vy, n_body);

    for (int i = 0; i < n_iteration; i++){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // TODO: MPI routine
        // MPI_Scatter
        update_velocity(local_m, local_x, local_y, local_vx, local_vy, n_body);
        update_position(local_x, local_y, local_vx, local_vy, n_body);
        // MPI_Gather
        // TODO End

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

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (rank == 0) {
		#ifdef GUI
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		glutInitWindowSize(500, 500); 
		glutInitWindowPosition(0, 0);
		glutCreateWindow("N Body Simulation MPI Implementation");
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glMatrixMode(GL_PROJECTION);
		gluOrtho2D(0, X_RESN, 0, Y_RESN);
		glutDisplayFunc(master);
        glutMainLoop();
        #else
        master();
		#endif
	} else {
        slave();
    }

	if (rank == 0){
		printf("Student ID: 119010001\n"); // replace it with your student id
		printf("Name: Your Name\n"); // replace it with your name
		printf("Assignment 2: N Body Simulation MPI Implementation\n");
	}

	MPI_Finalize();

	return 0;
}

