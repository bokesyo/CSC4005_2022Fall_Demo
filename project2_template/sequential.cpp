#define GLUT_DISABLE_ATEXIT_HACK 
 
#include <GL/glut.h>
#include <stdio.h>

int X_RESN, Y_RESN, total_size, max_iteration;
int* data;


typedef struct complextype {
    float real, imag;
} Compl;


void computation_sequential(int* data, int X_RESN, int Y_RESN, int max_iteration){
    /* Mandlebrot variables */
    int i, j, k;
    int count;
    Compl   z, c;
    float   lengthsq, temp;

    count = 0;
    for (i=0; i < X_RESN; i++) {
        for (j=0; j < Y_RESN; j++) {
            z.real = z.imag = 0.0;
			/* scale factors for X_RESN x Y_RESN window */
            c.real = ((float) j - X_RESN / 2)/(X_RESN / 2);               
            c.imag = ((float) i - Y_RESN / 2)/(Y_RESN / 2);
            k = 0;
            do { /* iterate for pixel color */
				temp = z.real*z.real - z.imag*z.imag + c.real;
				z.imag = 2.0*z.real*z.imag + c.imag;
				z.real = temp;
				lengthsq = z.real*z.real+z.imag*z.imag;
				k++;
            } while (lengthsq < 4.0 && k < max_iteration);

            if (k == max_iteration){
                data[count] = 1;
            } else {
                data[count] = 0;
            }
            count ++;
          }
    }
}


void Display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0f, 0.0f, 0.0f);
	
	GLfloat pointSize = 1.0f;
	glPointSize(pointSize);
	glBegin(GL_POINTS);
		glClear(GL_COLOR_BUFFER_BIT);
		glColor3f(1.0f, 0.0f, 0.0f);
		int x, y;
		int* pointer = data;
		for (x = 0; x < X_RESN; x++){
			for (y = 0; y < Y_RESN; y++) {
				pointer ++;
				if (*pointer == 1) {
					glVertex2f(x, y);
				}
			}
		}
	glEnd();
	glFlush();
	
}

void Initial()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, (float) X_RESN, 0.0, (float) Y_RESN);   // snapshot: left right bottom top
}


int main(int argc, char *argv[])
{
	X_RESN = 1000;
	Y_RESN = 1000;
	total_size = X_RESN * Y_RESN;
	max_iteration = 1000;
	data = new int[total_size];

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500); // this is the size of window, it has nothing to do with the coordinate of points. 
	glutCreateWindow("CSC4005"); // title
	glutDisplayFunc(Display);
	Initial();

	computation_sequential(data, X_RESN, Y_RESN, max_iteration);

	glutMainLoop();

	return 0;
}

