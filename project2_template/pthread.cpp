#include "asg2.h"
#include <stdio.h>
#include <pthread.h>

int n_thd;

typedef struct {
    //TODO: specify your arguments for threads
    int a;
    int b;
    //TODO END
} Args;


void* worker(void* args) {
    //TODO: procedure in each threads
    // the code following is not a necessary, you can replace it.
    
    /* Pass in arguments */
    Args* my_arg = (Args*) args;
    int a = my_arg->a;
    int b = my_arg->b;

    if (a == 0) {
        Point* p = data;
        for (int index = 0; index < total_size; index++){
            compute(p);
            p++;
        }
    }

    //TODO END

}


int main(int argc, char *argv[]) {

	if ( argc == 5 ) {
		X_RESN = atoi(argv[1]);
		Y_RESN = atoi(argv[2]);
		max_iteration = atoi(argv[3]);
        n_thd = atoi(argv[4]);
	} else {
		X_RESN = 1000;
		Y_RESN = 1000;
		max_iteration = 300;
        n_thd = 4;
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

    //TODO: assign jobs
    pthread_t thds[n_thd]; // thread pool
    Args args[n_thd]; // arguments for all threads
    for (int thd = 0; thd < n_thd; thd++){
        args[thd].a = thd;
        args[thd].b = n_thd;
    }
    for (int thd = 0; thd < n_thd; thd++) pthread_create(&thds[thd], NULL, worker, &args[thd]);
    for (int thd = 0; thd < n_thd; thd++) pthread_join(thds[thd], NULL);
    //TODO END
	/* computation part end */

	glutMainLoop();

	return 0;
}

