# CSC4005 Project 2 Template

Author: Bokai Xu

## We have a header file

Common functions and variables are included in `asg2.h` with some explaination.

## Where to run

This code can run on CSC4005 VM (both arm64 and x86_64 version).

Due to lack of GUI support of HPC cluster, please remove GUI (DIY) before you run experiment on cluster.


## Compile

```sh
g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm sequential.cpp -o sequential
```

```sh
mpic++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm mpi.cpp -o mpi
```

```sh
g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -lpthread pthread.cpp -o pthread
```


## This code helps you understand the whole picture.

```c++
#include <GL/glut.h>
#include <stdio.h>

/* define a struct called Compl to store information of a complex number*/
typedef struct complextype { float real, imag; } Compl;

/* define a struct called Point to store information of each point */
typedef struct pointtype { int x, y, color; } Point;

/*
X_RESN = resolution of x axis
Y_RESN = resolution of y axis
total_size = X_RESN * Y_RESN
max_iteration = a parameter of Mandelbrot computation
*/

int X_RESN, Y_RESN, total_size, max_iteration;

/* to store all the points, it will be initialized later */
Point* data;


void initData() {
	/*
	Intialize data storage.

	data = 
	|   x1   |   x2   |  ...  |   xn   |
	|   y1   |   y2   |  ...  |   yn   |
	| color1 | color2 |  ...  | colorn |

	it represents a 2D array where each entry has a color attribute.
	
	x_i is in {0, 1, ..., X_RESN)}
	y_i is in {0, 1, ..., Y_RESN)}
	color_i is in {0, 1}
	
	*/
	total_size = X_RESN * Y_RESN;
	data = new Point[total_size];
	int i, j;
	Point* p = data;
	for (i = 0; i < X_RESN; i++) {
        for (j = 0; j < Y_RESN; j++) {
			p->x = i;
			p->y = j;
			p ++;
		}
	}
}


void compute(Point* p) {
	/* 
	Give a Point p, compute its color.
	Mandelbrot.
	*/
	
	Compl z, c;
	float lengthsq, temp;
	int k;

	/* scale [0, X_RESN] x [0, Y_RESN] to [-1, 1] x [-1, 1] */
	c.real = ((float) p->x - X_RESN / 2) / (X_RESN / 2);
	c.imag = ((float) p->y - Y_RESN / 2) / (Y_RESN / 2);

	/* the following block is about math. */ 

	z.real = z.imag = 0.0;
    k = 0;

	do { 
		temp = z.real*z.real - z.imag*z.imag + c.real;
		z.imag = 2.0*z.real*z.imag + c.imag;
		z.real = temp;
		lengthsq = z.real*z.real+z.imag*z.imag;
		k++;
	} while (lengthsq < 4.0 && k < max_iteration);

	/* math block end */ 

	if (k == max_iteration){
		p->color = 1;
	} else {
		p->color = 0;
	}

}


void sequentialCompute(){
	Point* p;
	p = data;
	for (int index = 0; index < total_size; index++){
		compute(p);
		p++;
	}
}


void plot() {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0f, 0.0f, 0.0f);
	
	GLfloat pointSize = 1.0f;
	glPointSize(pointSize);
	glBegin(GL_POINTS);
		glClear(GL_COLOR_BUFFER_BIT);
		glColor3f(1.0f, 0.0f, 0.0f);
		
		int count;
		Point* p = data;
		for (count = 0; count < total_size; count++){
			if (p->color == 1) {
				glVertex2f(p->x, p->y);
			}
			p ++;
		}
	glEnd();
	glFlush();
	
}





int main(int argc, char *argv[]) {
	X_RESN = 1000;
	Y_RESN = 1000;
	max_iteration = 300;
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500); // this is the size of window, it has nothing to do with the coordinate of points. 
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Sequential");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, X_RESN, 0, Y_RESN); // snapshot: left right bottom top
	glutDisplayFunc(plot);

	/* computation part begin */
	initData();
	sequentialCompute();
	/* computation part end */

	glutMainLoop();

	return 0;
}


```