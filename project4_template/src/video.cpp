#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>


int n_body;
int bound_x;
int bound_y;
int n_iteration;
std::string version;


int main(int argc, char *argv[]){
    std::string path = argv[1];
    std::string meta_path = path + "/" + "metadata.txt";
    std::cout << "metadata path: " << meta_path << std::endl;
    std::ifstream f(meta_path);
    f >> version;
    f >> n_body;
    f >> bound_x;
    f >> bound_y;
    f >> n_iteration;
    std::cout << "version: " << version;
    std::cout << "n_body: " << n_body;
    std::cout << "n_iteration: " << n_iteration;
    f.close();

    std::string data_path = path + "/" + "data.txt";
    std::ifstream df(data_path);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, bound_x, 0, bound_y);

    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    for (int i = 0; i < n_iteration; i++){

        std::cout << "Iteration " << i << std::endl;

        for (int i = 0; i < n_body; i++){
            df >> x[i];
            df >> y[i];
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        double xi;
        double yi;
        for (int i = 0; i < n_body; i++){
            xi = x[i];
            yi = y[i];
            glVertex2f(xi, yi);
        }
        glEnd();
        glFlush();
        glutSwapBuffers();

    }

    sleep(5);
    
    return 0;

}


