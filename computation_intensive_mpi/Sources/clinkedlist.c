/* File     : clinkedlist.c
* Author    : Md Mokarrom Hossain
* Username  : x2013idf
* Course    : CSCI 522
* Purpose   : Tow linked lists(C-type) implementation for storing particles and collisions.
*/
#include <stdio.h>
#include "../Headers/clinkedlist.h"

void InitParticleList (particleList_t *pList)
{
    pList->head = NULL;
}

particleListItem_t* CreateParticleListItem(float x, float y, float vx, float vy)
{
    particleListItem_t* newPartItem = (particleListItem_t*) malloc (sizeof (particleListItem_t) );

    newPartItem->part.pcord.x = x;
    newPartItem->part.pcord.y = y;
    newPartItem->part.pcord.vx = vx;
    newPartItem->part.pcord.vy = vy;
    newPartItem->part.ptype = 0;

    return newPartItem;
}

void InsertPartListFront(particleList_t *pList, particleListItem_t *partItem)
{
    partItem->next = pList->head;
    pList->head = partItem;
}

void RemoveParticle(particleList_t *pList, particleListItem_t *removedItem)
{
    particleListItem_t *tmp = pList->head;

    if (tmp != NULL && tmp == removedItem)// Deleting the first item.
    {
        pList->head = pList->head->next;
        tmp->next = NULL;   //NULL out what the removed node points to
        free(removedItem);
        return;
    }

    while (tmp->next != NULL && tmp->next != removedItem)
        tmp = tmp->next;

    if (tmp->next != NULL)
    {
        tmp->next = removedItem->next;
        removedItem->next = NULL;   //NULL out what the removed node points to
        free(removedItem);
    }
}

int length(particleList_t *pList)   // returns the length of particle list.
{
    particleListItem_t *aParticle  = pList->head;
    int count = 0;

    while( aParticle != NULL)
    {
        count++;
        aParticle = aParticle->next;
    }
    return count;
}

void ClearParticleList(particleList_t *pList)
{
    particleListItem_t *curParticle = NULL, *tmp = NULL;
    curParticle = pList->head;

    while( curParticle != NULL)
    {
        tmp = curParticle->next;
        curParticle->next = NULL;
        free(curParticle);
        curParticle = tmp;
    }
    pList->head = NULL;
}

void PrintParticles(particleList_t *pList)
{
    printf("===========Printing the particle list=========================\n");
    particleListItem_t *curParticle  = pList->head;
    while( curParticle != NULL)
    {
        pcord_t pcord = (curParticle->part.pcord);
        printf("%f     %f     %f     %f\n", pcord.x, pcord.y, pcord.vx, pcord.vy);
        //particle_t *p = &(curParticle->part);
        //printf("y = %lf     \n", p->pcord.y);
        curParticle = curParticle->next;
    }
    printf("===========End particle list=========================\n");
}

void InitCollisionList (collisionList_t *cList)
{
    cList->head = NULL;
}

collisionListItem_t* CreateCollisionListItem(particle_t part1, particle_t part2, float t)
{
    collisionListItem_t* newCollisionPair = (collisionListItem_t*) malloc (sizeof (collisionListItem_t) );

    newCollisionPair->part1 = part1;
    newCollisionPair->part2 = part2;
    newCollisionPair->t = t;

    return newCollisionPair;
}

void InsertCollListFront(collisionList_t *cList, collisionListItem_t *cItem)
{
    cItem->next = cList->head;
    cList->head = cItem;
}

void ClearCollisionList(collisionList_t *cList)
{
    collisionListItem_t *curCollision = NULL, *tmp = NULL;
    curCollision = cList->head;

    while( curCollision != NULL)
    {
        tmp = curCollision;
        curCollision = curCollision->next;
        tmp->next = NULL;
        free(tmp);
    }
    cList->head = NULL;
}
