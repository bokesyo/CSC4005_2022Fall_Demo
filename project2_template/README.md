# CSC4005 Project 2 Template

This code can run on CSC4005 VM (both arm64 and x86_64 version). You have to run it using GUI Desktop of your VM.

This code can also run on HPC cluster (only command line, no GUI, see instructions below). 


![](gui_mpi.png)


# Description

The template includes the following component:

- Sequential version is completed for your reference.
- MPI version and pthread version are not completed.

To do parallelization, you have multiple choices. You are encouraged to use some brand new method to partition the data. 


Header files: `asg2.h`. Common functions and variables are included in `asg2.h` with some explaination.


Source code: `sequential.cpp`, `pthread.cpp`, `mpi.cpp`.


# Compile

### Sequential without GUI (completed, for reference)
```sh
g++ sequential.cpp -o sequential -std=c++11 -O2
```

### Sequential with GUI (completed, for reference)
```sh
g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm sequential.cpp -o sequential -std=c++11 -DGUI -O2
```

### MPI without GUI (finish #TODO by yourself)
```sh
mpic++ mpi.cpp -o mpi -std=c++11
```

### MPI with GUI (finish `#TODO` by yourself)
```sh
mpic++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm mpi.cpp -o mpi -std=c++11 -DGUI
```


### pthread without GUI (finish `#TODO` by yourself)
```sh
g++ pthread.cpp -o pthread -std=c++11 -O2
```

### pthread with GUI (finish `#TODO` by yourself)
```sh
g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -lpthread pthread.cpp -o pthread -std=c++11 -DGUI -O2
```

## About `#ifdef GUI` and `-DGUI`

This command is to control if the compiler should output a GUI application. To enable it, use `gcc xxxx -DGUI` to let compiler know it should output a GUI application. To disable it, just omit `-DGUI` so the compiler will output a command line application.

## About `-O2`

This argument is a optimization method for your program. Why we want to use it among all versions (`mpi`, `pthread`, `sequential`)? Because `mpic++` by default utilize `O2` optimization. It is not fair for `sequential` and `pthread` version. 

## About usage of `-I -L -l` 

Please refer to Tutorial 1. We have talked about it.


# Run

`X_RESN` means the resolution of x axis, `Y_RESN` means the resolution of y axis.

`max_iteration` is a parameter of Mandelbrot Set computation.

`n_proc` is the number of processed of MPI.

`n_thd` is the number of threads of pthread.

### Sequential
```sh
./sequential $X_RESN $Y_RESN $max_iteration
```

### MPI
```sh
mpirun -np $n_proc ./mpi $X_RESN $Y_RESN $max_iteration
```

### pthread
```sh
./pthread $X_RESN $Y_RESN $max_iteration $n_thd
```

If you choose to build a GUI application, you should see a window as well when you execute it.


# Check the correctness of your result

`Not implemented.`



# Run your job on HPC cluster

For mpi program, you can 



# This code helps you understand the whole picture.

```c++
#include <stdio.h>
#include <GL/glut.h>


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
			// p->color = 1;
			p ++;
		}
	}
}


void compute(Point* p) {
	/* 
	Give a Point p, compute its color.
	Mandelbrot Set Computation.
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


void plot() {
	/*
	Plot all the points to screen.
	*/

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


void sequentialCompute(){
	/* compute for all points one by one */
	Point* p = data;
	for (int index = 0; index < total_size; index++){
		compute(p);
		p++;
	}
}


int main(int argc, char *argv[]) {
	if ( argc == 3 ) {
		X_RESN = (int) argv[0];
		Y_RESN = (int) argv[1];
		max_iteration = (int) argv[2];
	} else {
		X_RESN = 1000;
		Y_RESN = 1000;
		max_iteration = 300;
	}
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Sequential");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, X_RESN, 0, Y_RESN);
	glutDisplayFunc(plot);

	/* computation part begin */
	initData();
	sequentialCompute();
	/* computation part end */

	glutMainLoop();

	return 0;
}




```


Any questions about this template, please contact Bokai Xu.
