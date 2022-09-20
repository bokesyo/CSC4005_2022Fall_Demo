/* File     : physics.c
* Purpose   : Supporting functions(physics) implementation for particle simulation.
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/physics.h"

#ifndef sqr
#define sqr(a) ((a)*(a))
#endif

#ifndef sign
#define sign(a) ((a) > 0 ? 1 : -1)
#endif

int feuler(pcord_t *a, float time_step){
    a->x = a->x + time_step* a->vx ;
    a->y = a->y + time_step* a->vy ;
    return 0 ;
}

float wall_collide(pcord_t *p, cord_t wall){
    float gPreassure = 0.0 ;

    if(p->x < wall.x0){
        p->vx = -p->vx ;
        p->x  = wall.x0 + (wall.x0 - p->x);
        gPreassure += 2.0 * fabs(p->vx);
    }
    if(p->x > wall.x1){
        p->vx = -p->vx ;
        p->x  = wall.x1 - (p->x - wall.x1);
        gPreassure += 2.0 * fabs(p->vx);
    }
    if(p->y < wall.y0){
        p->vy = -p->vy ;
        p->y  = wall.y0 + (wall.y0 - p->y);
        gPreassure += 2.0 * fabs(p->vy);
    }
    if(p->y > wall.y1){
        p->vy = -p->vy ;
        p->y  = wall.y1 - (p->y - wall.y1);
        gPreassure += 2.0 * fabs(p->vy);
    }
    return gPreassure ;
}

float collide(pcord_t *p1, pcord_t *p2){
    double a, b, c;
    double temp, t1, t2;

    a = sqr(p1->vx - p2->vx) + sqr(p1->vy - p2->vy);
    b = 2 * ((p1->x - p2->x) * (p1->vx - p2->vx) + (p1->y - p2->y) * (p1->vy - p2->vy));
    c = sqr(p1->x - p2->x) + sqr(p1->y - p2->y) - 4*1*1;

    if (a != 0.0){
        temp = sqr(b) - 4*a*c;
        if (temp >= 0){
            temp = sqrt(temp);
            t1=(-b + temp) / (2*a);
            t2=(-b - temp) / (2*a);

            if (t1 > t2){
                temp = t1;
                t1 = t2;
                t2 = temp;
            }
            if ((t1 >= 0)&(t1 <= 1))
                return t1;
            else if ((t2>=0)&(t2<=1))
                return t2;
        }
    }
    return -1;
}

void interact(pcord_t *p1, pcord_t *p2, float t){
    float c, s, a, b, tao;
    pcord_t p1temp, p2temp;

    if (t >= 0){

        /* Move to impact point */
        (void) feuler (p1, t);
        (void) feuler (p2, t);

        /* Rotate the coordinate system around p1 */
        p2temp.x = p2->x - p1->x;
        p2temp.y = p2->y - p1->y;

        /* Givens plane rotation, Golub, van Loan p. 216 */
        a = p2temp.x;
        b = p2temp.y;

        if (p2->y == 0){
            c = 1; s = 0;
        }
        else{
            if (fabs(b) > fabs(a)){
            tao = -a/b;
            s = 1 / (sqrt(1 + sqr(tao)));
            c = s * tao;
            }
            else{
            tao = -b/a;
            c = 1 / (sqrt(1 + sqr(tao)));
            s = c * tao;
            }
        }

        p2temp.x = c * p2temp.x + s * p2temp.y; /* This should be equal to 2r */
        p2temp.y = 0.0;

        p2temp.vx = c * p2->vx + s * p2->vy;
        p2temp.vy = -s * p2->vx + c * p2->vy;
        p1temp.vx = c * p1->vx + s * p1->vy;
        p1temp.vy = -s * p1->vx + c * p1->vy;

        /* Assume the balls has the same mass... */
        p1temp.vx = -p1temp.vx;
        p2temp.vx = -p2temp.vx;

        p1->vx = c * p1temp.vx - s * p1temp.vy;
        p1->vy = s * p1temp.vx + c * p1temp.vy;
        p2->vx = c * p2temp.vx - s * p2temp.vy;
        p2->vy = s * p2temp.vx + c * p2temp.vy;

        /* Move the balls the remaining time. */
        c = 1.0 - t;
        (void) feuler (p1, c);
        (void) feuler (p2, c);
    }
}

////////For handling big particle (bonus project)

void wall_collide_Big(pcord_t *p, cord_t wall, float rBig){

    if(p->x < wall.x0+rBig){
	p->vx = -p->vx ;
	p->x  = wall.x0 + rBig + (wall.x0+rBig - p->x);
    }
    if(p->x > wall.x1-rBig){
	p->vx = -p->vx ;
	p->x  = wall.x1-rBig - (p->x- (wall.x1-rBig));
    }
    if(p->y < wall.y0+rBig){
	p->vy = -p->vy ;
	p->y  = wall.y0 + rBig + (wall.y0+rBig - p->y);
    }
    if(p->y > wall.y1-rBig){
	p->vy = -p->vy ;
	p->y  = wall.y1-rBig - (p->y-(wall.y1-rBig));
    }
}

float collideBig(pcord_t *Big, pcord_t *p, float rBig){
    double a,b,c;
    double temp,t1,t2;

    a=sqr(p->vx)+sqr(p->vy);
    b=2*((p->x - Big->x)*(p->vx)+(p->y - Big->y)*(p->vy));
    c=sqr(Big->x-p->x)+sqr(Big->y-p->y)-sqr(1+rBig);

    if (a!=0.0){
	temp=sqr(b)-4*a*c;
	if (temp>=0){
	    temp=sqrt(temp);
	    t1=(-b+temp)/(2*a);
	    t2=(-b-temp)/(2*a);

	    if (t1>t2){
		temp=t1;
		t1=t2;
		t2=temp;
	    }
	    if ((t1>=0)&(t1<=1))
		return t1;
	    else if ((t2>=0)&(t2<=1))
		return t2;
	}
    }
    return -1;
}

void interactBig(pcord_t *Big, float massBig, pcord_t *p, float t){

    float c,s,a1,b1,c1,aa,bb,tao,v1,temp;
    pcord_t p1temp,p2temp;

    if (t>=0){

	/* Move little ball to impact point */
	(void)feuler(p,t);

	/* Rotate the coordinate system around the Big Ball*/
	p1temp.x=p->x-Big->x;
	p1temp.y=p->y-Big->y;

	/* Givens plane rotation, Golub, van Loan p. 216 */
	a1=p1temp.x;
	b1=p1temp.y;
	if (p->y==0){
	    c=1;s=0;
	}
	else{
	    if (fabs(b1)>fabs(a1)){
		tao=-a1/b1;
		s=1/(sqrt(1+sqr(tao)));
		c=s*tao;
	    }
	    else{
		tao=-b1/a1;
		c=1/(sqrt(1+sqr(tao)));
		s=c*tao;
	    }
	}

	p1temp.x=c * p1temp.x+s * p1temp.y;
	/* This should be equal to r+rBig */
	p1temp.y=0.0;

	p1temp.vx= c* p->vx + s* p->vy;
	p1temp.vy=-s* p->vx + c* p->vy;
	p2temp.vx= c* Big->vx + s* Big->vy;
	p2temp.vy=-s* Big->vx + c* Big->vy;

	/* Adjust the velocities of the particles */
	bb=massBig*sqr(p2temp.vx)+sqr(p1temp.vx);
	aa=massBig*p2temp.vx+p1temp.vx;
	a1=massBig+1.0;
	b1=-2.0*aa;
	c1=aa*aa-massBig*bb;

	if (a1!=0.0){
	    temp=sqr(b1)-4*a1*c1;
	    if (temp>=0){
		temp=sqrt(temp);
		if (b1<0)
		    v1=(-b1+temp)/(2*a1);
		else
		    v1=-2*c/(b1+temp);
		if (fabs(v1-p2temp.vx)>1e-8){
		    if (b1>0)
			v1=(-b1-temp)/(2*a1);
		    else
			v1=-2*c/(b1-temp);
		}
	    }
	}

	p1temp.vx=v1;
	p2temp.vx=(aa-v1)/massBig;

	Big->vx = c * p2temp.vx - s * p2temp.vy;
	Big->vy = s * p2temp.vx + c * p2temp.vy;
	p->vx = c * p1temp.vx - s * p1temp.vy;
	p->vy = s * p1temp.vx + c * p1temp.vy;

	/* Move the little ball the remaining time. */
	c=1.0-t;
	(void)feuler(p,c);
    }
}




