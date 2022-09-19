/* File     : physics.h
* Purpose   : Supporting functions(physics) declaration for particle simulation.
*/
#ifndef _physics_h
#define _physics_h

#include "coordinate.h"

#define STEP_SIZE 1.0 /* the step size use in the integration */

int feuler(pcord_t *a, float time_step) ;

float wall_collide(pcord_t *p, cord_t wall) ;

float collide(pcord_t *p1, pcord_t *p2) ;

void interact(pcord_t *p1, pcord_t *p2, float t);

//For handling big particle (bonus project)
void wall_collide_Big(pcord_t *p, cord_t wall, float rBig) ;

float collideBig(pcord_t *Big, pcord_t *p, float rBig) ;

void interactBig(pcord_t *Big, float massBig, pcord_t *p, float t) ;

#endif
