#include <stdio.h>
#include <cstdlib>
#include <chrono>
#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif


/* define a struct called Compl to store information of a complex number*/
typedef struct complextype { float real, imag; } Compl;

/* define a struct called Point to store information of each point */
typedef struct pointtype { int x, y; float color; } Point;

/*
X_RESN = resolution of x axis
Y_RESN = resolution of y axis
total_size = X_RESN * Y_RESN
max_iteration = a parameter of Mandelbrot computation
*/
int X_RESN, Y_RESN, total_size, max_iteration;

/* to store all the points, it will be initialized later */
Point* data;

/* to keep track of time */
std::chrono::high_resolution_clock::time_point t1;
std::chrono::high_resolution_clock::time_point t2;
std::chrono::duration<double> time_span;


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
	It is not necessary to modify this function, because it is a completed one.
	*** However, to further improve the performance, you may change this function to do batch computation.
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

	p->color = (float) k / max_iteration;

}


#ifdef GUI
void plot() {
	/* Plot all the points to screen. */

	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0f, 0.0f, 0.0f);
	
	GLfloat pointSize = 1.0f;
	glPointSize(pointSize);
	glBegin(GL_POINTS);
		glClear(GL_COLOR_BUFFER_BIT);
		
		int count;
		Point* p = data;
		for (count = 0; count < total_size; count++){
			glColor3f(1.0f-p->color, 1.0f-p->color, 1.0f-p->color); // control the color
			glVertex2f(p->x, p->y); // plot a point
			p ++;
		}
		
	glEnd();
	glFlush();
	
}

#endif


void sequentialCompute(){
	/* compute for all points one by one */
	Point* p = data;
	for (int index = 0; index < total_size; index++){
		compute(p);
		p++;
	}
}


int main(int argc, char *argv[]) {
	/* pass in metadata for computation */
	if ( argc == 4 ) {
		X_RESN = atoi(argv[1]);
		Y_RESN = atoi(argv[2]);
		max_iteration = atoi(argv[3]);
	} else {
		X_RESN = 1000;
		Y_RESN = 1000;
		max_iteration = 100;
	}

	#ifdef GUI
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Sequential");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, X_RESN, 0, Y_RESN);
	glutDisplayFunc(plot);
	#endif

	/* computation part begin */
    t1 = std::chrono::high_resolution_clock::now();
	initData();
	sequentialCompute();
	t2 = std::chrono::high_resolution_clock::now();  
	time_span = t2 - t1;
	/* computation part end */

	printf("Student ID: 119010001\n"); // replace it with your student id
	printf("Name: Your Name\n"); // replace it with your name
	printf("Assignment 2 Sequential\n");
	printf("Run Time: %f seconds\n", time_span.count());
	printf("Problem Size: %d * %d, %d\n", X_RESN, Y_RESN, max_iteration);
	printf("Process Number: %d\n", 1);

	#ifdef GUI
	glutMainLoop();
	#endif

	return 0;
}

