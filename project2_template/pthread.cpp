#include "asg2.h"
#include <stdio.h>
#include <pthread.h>


typedef struct {
    //TODO: specify your arguments for threads
    int a;
    int b;
} Args;


void* worker(void* args) {
    Args* my_arg = (Args*) args;
    
    //TODO: procedure in each threads
    // printf("Thread Id:%d", pthread_self());

    // let me do nothing...
}


int main(int argc, char *argv[]) {
    int n_thd = 4;

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

    pthread_t thds[n_thd]; // thread pool
    Args args[n_thd]; // arguments for all threads

    //TODO: 
    for (int thd = 0; thd < n_thd; thd++){
        args[thd].a = 0;
        args[thd].b = 1;
    }

    for (int thd = 1; thd < n_thd; thd++){
        pthread_create(&thds[thd], NULL, worker, &args[thd]);
        printf("Thread %d/%d Start Working..\n", thd, n_thd);
    }

    for (int thd = 1; thd < n_thd; thd++){
        pthread_join(thds[thd], NULL);
    }
	/* computation part end */

	glutMainLoop();

	return 0;
}

