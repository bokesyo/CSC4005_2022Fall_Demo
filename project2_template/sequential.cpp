#include "asg2.h"
#include <stdio.h>


void sequentialCompute(){
	/* compute for all points one by one */
	Point* p = data;
	for (int index = 0; index < total_size; index++){
		compute(p);
		p++;
	}
}


int main(int argc, char *argv[]) {
	X_RESN = 1000;
	Y_RESN = 1000;
	max_iteration = 300;
	
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

