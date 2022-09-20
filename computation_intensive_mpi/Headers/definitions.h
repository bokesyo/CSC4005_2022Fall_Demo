/* File     : definitions.h
* Purpose   : MACRO definition for particle simulation.
*/
#include<stdlib.h>
#include<math.h>

#include "coordinate.h"

#ifndef _definitions_h
#define _definitions_h

#define PI 3.141592653

#define MAX_NO_PARTICLES  10000  /* Maximum number of particles/processor */
#define INIT_NO_PARTICLES 500    /* Initial number of particles/processor */
#define MAX_INITIAL_VELOCITY 50

#define BOX_HORIZ_SIZE 10000.0
#define BOX_VERT_SIZE 10000.0
#define WALL_LENGTH (2.0*BOX_HORIZ_SIZE + 2.0*BOX_VERT_SIZE)

#define PARTICLE_BUFFER_SIZE MAX_NO_PARTICLES/5
#define COMM_BUFFER_SIZE PARTICLE_BUFFER_SIZE

typedef int bool;
enum { false, true };

struct particle {
  pcord_t  pcord;
  int ptype;        /* Used to simulate mixing of gases */
};

typedef struct particle particle_t;

struct collision{
  particle_t p1, p2;
  float t;
};

typedef struct collision collision_t;

#endif
