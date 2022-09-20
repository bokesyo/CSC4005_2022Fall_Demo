/* File     : clinkedlist.h
* Author    : Md Mokarrom Hossain
* Username  : x2013idf
* Course    : CSCI 522
* Purpose   : Tow linked lists(C-type) declaration for storing particles and collisions.
*/
#ifndef CLINKEDLIST_H_INCLUDED
#define CLINKEDLIST_H_INCLUDED

#include "definitions.h"

struct particleListItem{
    particle_t part;
    struct particleListItem *next;
};

typedef struct particleListItem particleListItem_t;

struct particleList{
    particleListItem_t *head;
};

typedef struct particleList particleList_t;

struct collisionListItem{
    particle_t part1, part2;
    float t;
    struct collisionListItem *next;
};

typedef struct collisionListItem collisionListItem_t;

struct collisionList{
    collisionListItem_t *head;
};

typedef struct collisionList collisionList_t;

void InitParticleList (particleList_t *pList);            // initialize an empty particle list.
particleListItem_t* CreateParticleListItem(float x, float y, float vx, float vy);
void InsertPartListFront(particleList_t *pList, particleListItem_t *partItem);
void RemoveParticle(particleList_t *pList, particleListItem_t *removedItem);
int length(particleList_t *pList);
void ClearParticleList(particleList_t *pList);
void PrintParticles(particleList_t *pList);

void InitCollisionList (collisionList_t *cList);
collisionListItem_t* CreateCollisionListItem(particle_t part1, particle_t part2, float t);
void InsertCollListFront(collisionList_t *cList, collisionListItem_t *cItem);
void ClearCollisionList(collisionList_t *cList);

#endif // CLINKEDLIST_H_INCLUDED
