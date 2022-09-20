/* File: mpiPartSim.c
* -------------------------------------------------
* Name      : Md Mokarrom Hossain
* Username  : x2013idf
* Course    : CSCI 522
* Purpose   : MPI based parallel implementation of particle simulation.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "../Headers/physics.h"
#include "../Headers/definitions.h"
#include "../Headers/clinkedlist.h"
#include "mpi.h"

#define TOTAL_TIME 100
#define TIME_STEP   1
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

float FloatRand(float fMin, float fMax);
int IsOutsideLocalBox(pcord_t *p, cord_t box);
void AddToSendBuffer(particle_t *sendBudAdd, particle_t *p);
void CalculateGridLayout(int nProcs, int *gridRows, int *gridCols);

int main(int argc, char ** argv)
{
    double lPressure=0.0, gPressure=0.0, lTemp=0.0, gTemp=0.0, V=0.0/* Volume */, R=0.0/* magic constant */;
    float bigPartRadius = 100.0, bigPartMass = 1000.0;
    unsigned int gNumParts = 20000, lNumParts = 0;
    float gBoxWidth = BOX_HORIZ_SIZE, gBoxHeight = BOX_VERT_SIZE, lBoxWidth, lBoxHeight, max_vel = MAX_INITIAL_VELOCITY;
    double startTime = 0.0, endTime = 0.0;

    particleList_t particlesList, collideBigPartList;
    particleListItem_t *p1 = NULL, *p2 = NULL;
    collisionList_t collisionsList;
    collisionListItem_t *c = NULL;

    cord_t globalBox, localBox;
    particle_t *bigPart, sendBuffer[4][COMM_BUFFER_SIZE], recvBuffer[4][COMM_BUFFER_SIZE];
    int rank, noProcs, gridRows, gridCols, reorder = 1, neighbours[4], sendCounts[4]={0}, dims[2], myCoords[2], periods[2] = {1, 1};

    MPI_Comm gridComm;
    MPI_Request request[4];
    MPI_Status status[4];

    globalBox.x0 = 0;
    globalBox.y0 = 0;
    globalBox.x1 = gBoxWidth;
    globalBox.y1 = gBoxHeight;

    //starts MPI
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &noProcs);

    //Create MPI custom type in order to ease the communication.
    MPI_Datatype mpiPartType, oldTypes[2] = {MPI_FLOAT, MPI_INT};
    int blockcounts[2] = {4, 1};
    MPI_Aint offsets[2], extent;

    offsets[0] = 0;
    MPI_Type_extent(MPI_FLOAT, &extent);
    offsets[1] = 4 * extent;

    MPI_Type_struct(2, blockcounts, offsets, oldTypes, &mpiPartType);
    MPI_Type_commit(&mpiPartType);

    srand((unsigned int) time(NULL) + rank*100);    //randomized the seed.

    CalculateGridLayout(noProcs, &gridRows, &gridCols);
    dims[0] = gridRows;
    dims[1] = gridCols;
    MPI_Dims_create(noProcs, 2, dims); //Creates a division of processors in 2-D cartesian grid.

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &gridComm);
    MPI_Cart_get(gridComm, 2, dims, periods, myCoords); // myCoords[0] = row, myCoords[0] = col.

    //calculate the local box dimension.
    lBoxWidth = gBoxWidth / gridCols;
    lBoxHeight = gBoxHeight / gridRows;
    localBox.x0 = myCoords[1] * lBoxWidth;
    localBox.y0 = myCoords[0] * lBoxHeight;
    localBox.x1 = (myCoords[1] == gridCols-1) ? gBoxWidth : (myCoords[1]+1) * lBoxWidth;
    localBox.y1 = (myCoords[0] == gridRows-1) ? gBoxHeight : (myCoords[0]+1) * lBoxHeight;

    //Get the rank of neighbor processors.
    MPI_Cart_shift(gridComm, 0, 1, &neighbours[DOWN], &neighbours[UP]);
    MPI_Cart_shift(gridComm, 1, 1, &neighbours[LEFT], &neighbours[RIGHT]);

    if (rank == 0)
    {
        printf("\nParticle Simulation: Ideal Gas Law(PV = nRT) verification.\n");fflush(stdout);
        printf("------------------------------------------------------------\n");fflush(stdout);
        printf("# processors = %d \t # particles = %d\n", noProcs, gNumParts);fflush(stdout);
        printf("Cartesian Grid = %dX%d \t total time-steps = %d\n", dims[0], dims[1], TOTAL_TIME);fflush(stdout);
        printf("Box: width=%.0lf height=%.0lf\tBig Particle: radius=%.0lf mass=%.0lf\n\n", gBoxWidth, gBoxHeight, bigPartRadius, bigPartMass);fflush(stdout);
        startTime = MPI_Wtime();    //Starts the timer.
    }

    if (rank == noProcs - 1){
        lNumParts = gNumParts - (rank*(gNumParts/noProcs));
    }
    else{
        lNumParts = gNumParts / noProcs;
    }

    InitParticleList(&particlesList);
    int i, j, k, dir;
    bool bFlag, hasBigPart;

    // Initiate particles.
    for(i = 0; i < lNumParts; i++)
    {
        float x = FloatRand(localBox.x0, localBox.x1);
        float y = FloatRand(localBox.y0, localBox.y1);
        float r = FloatRand(0, max_vel);
        float theta = FloatRand(0, 2 * PI);
        float vx = r * cos(theta);
        float vy = r * sin(theta);
        InsertPartListFront(&particlesList, CreateParticleListItem(x, y, vx, vy));
        lTemp += (double)(r * r) / 2;   // Here r = v = velocity.
    }
    //initiate big particle
    bigPart = (particle_t*) malloc (sizeof (particle_t) );
    hasBigPart = false;
    if (rank == 0)
    {
        bigPart->pcord.x = FloatRand(globalBox.x0, globalBox.x1);
        bigPart->pcord.y = FloatRand(globalBox.y0, globalBox.y1);
        float r = FloatRand(0, max_vel);
        float theta = FloatRand(0, 2 * PI);
        bigPart->pcord.vx = r * cos(theta);
        bigPart->pcord.vy = r * sin(theta);
        bigPart->ptype = 100;
    }

    MPI_Bcast(bigPart, 1, mpiPartType, 0, gridComm);
    if (IsOutsideLocalBox(&(bigPart->pcord), localBox) == -1)
        hasBigPart = true;
    else
        hasBigPart = false;

    MPI_Reduce(&lTemp, &gTemp, 1, MPI_DOUBLE, MPI_SUM, 0, gridComm);
    if (rank == 0)  gTemp = gTemp / (double)gNumParts;

    InitCollisionList(&collisionsList);
    InitParticleList(&collideBigPartList);

    for (i = 0; i < TOTAL_TIME; i++)
    {//printf("time-step = %d rank = %d is running....\n",i, rank);fflush(stdout);
        for(j = 0; j < 4; j++)
        {
            sendCounts[j] = 0;
        }
        //Step # 1 : Check for collisions with big particle.
        if(hasBigPart == true)
        {
            p1 =  particlesList.head;
            while (p1 != NULL)
            {
                float t = collideBig(&(bigPart->pcord), &(p1->part.pcord), bigPartRadius);
                if(t == -1) // No collision with big particle.
                {
                    p1 = p1->next;
                }
                else    //Collides with big particle.
                {
                    particleListItem_t *tmp = NULL;
                    interactBig(&(bigPart->pcord), bigPartMass, &(p1->part.pcord), t);
                    InsertPartListFront(&collideBigPartList, CreateParticleListItem(p1->part.pcord.x, p1->part.pcord.y, p1->part.pcord.vx, p1->part.pcord.vy));
                    tmp = p1->next;
                    RemoveParticle(&particlesList, p1);
                    p1 = tmp;
                }
            }//End while loop.
            feuler(&(bigPart->pcord), TIME_STEP);    //move the big particle.
            wall_collide_Big(&(bigPart->pcord), globalBox, bigPartRadius);  //Check for wall collision with big particle.
            dir = IsOutsideLocalBox(&(bigPart->pcord), localBox);
            if (dir == -1)  // big particle is in my region(local box).
            {
                hasBigPart = true;
            }
            else    //big particle is outside my region(local box).
            {
                hasBigPart = false;
                AddToSendBuffer(&(sendBuffer[dir][sendCounts[dir]++]), bigPart);
            }
        }//end of big part branch.

        //Step # 2 : Check for collisions between the remaining unmoved particles.
        p1 =  particlesList.head;
        while (p1 != NULL)
        {
            bFlag = true;
            p2 = p1->next;
            while (p2 != NULL)
            {
                float t = collide(&(p1->part.pcord), &(p2->part.pcord));
                if (t == -1)    //No collision between p1 & p2.
                {
                    p2 = p2->next;
                }
                else    //p1 & p2 collides each other.
                {
                    particleListItem_t *tmp = NULL;
                    InsertCollListFront(&collisionsList, CreateCollisionListItem(p1->part, p2->part, t));
                    tmp = (p1->next == p2) ? p2->next : p1->next;
                    RemoveParticle(&particlesList, p1);
                    RemoveParticle(&particlesList, p2);
                    p1 = tmp;   bFlag = false;
                    break;
                }
            }//End inner while loop.
            if (bFlag == true)      p1 = p1->next;
        }//End outer while loop.

        //Step # 3 : Move particles that has not collided with another.
        p1 =  particlesList.head;
        while (p1 != NULL)
        {
            feuler(&(p1->part.pcord), TIME_STEP);
            p1 = p1->next;
        }//End while loop.

        //Step # 3 : Interact the colliding particles.
        c = collisionsList.head;
        while (c != NULL)
        {
            interact(&(c->part1.pcord), &(c->part2.pcord), c->t);
            InsertPartListFront(&particlesList, CreateParticleListItem(c->part1.pcord.x, c->part1.pcord.y, c->part1.pcord.vx, c->part1.pcord.vy));
            InsertPartListFront(&particlesList, CreateParticleListItem(c->part2.pcord.x, c->part2.pcord.y, c->part2.pcord.vx, c->part2.pcord.vy));
            c = c->next;
        }//End while loop.
        ClearCollisionList(&collisionsList); //free the memory occupied by collision list.

        //Again add those who collided with big particle.
        p1 =  collideBigPartList.head;
        while (p1 != NULL)
        {
            InsertPartListFront(&particlesList, CreateParticleListItem(p1->part.pcord.x, p1->part.pcord.y, p1->part.pcord.vx, p1->part.pcord.vy));
            p1 = p1->next;
        }//End while loop.
        ClearParticleList(&collideBigPartList);

        //Step # 4 : Check for wall interaction and add the momentum.
        p1 =  particlesList.head;
        while (p1 != NULL)
        {
            lPressure += (double)wall_collide(&(p1->part.pcord), globalBox);
            //Check whether particle is outside the local box.
            dir = IsOutsideLocalBox(&(p1->part.pcord), localBox);

            if (dir == -1)  //Particle(p1) is inside the local box.
            {
                p1 = p1->next;
            }
            else    //Particle(p1) is outside i.e. in neighbor region.
            {
                particleListItem_t *tmp = NULL;
                AddToSendBuffer(&(sendBuffer[dir][sendCounts[dir]++]), &(p1->part));
                tmp = p1->next;
                RemoveParticle(&particlesList, p1); // Remove neighbor's particle from my particle list.
                p1 = tmp;
            }
        }//End while.

        //Send neighbor's data to the corresponding neighbor.
        for(j = 0; j < 4; j++){
          MPI_Isend(sendBuffer[j], sendCounts[j], mpiPartType, neighbours[j], 0, gridComm, &(request[j]));
        }
        //Receive own data from neighbor.
        for(j = 0; j < 4; j++){
            MPI_Recv(recvBuffer[j], COMM_BUFFER_SIZE, mpiPartType, MPI_ANY_SOURCE, 0, gridComm, &(status[j]));
        }
        MPI_Waitall(4, request, MPI_STATUS_IGNORE); //Wait for non-blocking send completion.

        for(j = 0; j < 4; j++)
        {
            int noRecvdPart;
            MPI_Get_count(&(status[j]), mpiPartType, &noRecvdPart);
            for(k = 0; k < noRecvdPart; k++)
            {
                if (recvBuffer[j][k].ptype == 100)  //Received big particle.
                {
                    bigPart->pcord.x = recvBuffer[j][k].pcord.x;
                    bigPart->pcord.vx = recvBuffer[j][k].pcord.vx;
                    bigPart->pcord.y = recvBuffer[j][k].pcord.y;
                    bigPart->pcord.vy = recvBuffer[j][k].pcord.vy;
                    bigPart->ptype = recvBuffer[j][k].ptype;
                    hasBigPart = true;
                }
                else
                {
                    InsertPartListFront(&particlesList, CreateParticleListItem(recvBuffer[j][k].pcord.x, recvBuffer[j][k].pcord.y, recvBuffer[j][k].pcord.vx, recvBuffer[j][k].pcord.vy));
                }
            }//End inner for loop.
        }//End outer for loop.
    }// End time step for loop.
    ClearParticleList(&particlesList);
    free(bigPart);

    //printf("Rank=%d (%d,%d) : # particles=%d region=(%.1f,%.1f) (%.1f,%.1f)\n", rank, myCoords[0], myCoords[1], lNumParts, localBox.x0, localBox.y0, localBox.x1, localBox.y1);fflush(stdout);

    MPI_Reduce(&lPressure, &gPressure, 1, MPI_DOUBLE, MPI_SUM, 0, gridComm);
    if(rank == 0)
    {
        endTime = MPI_Wtime();
        double gBoxPerimeter = 2 * (gBoxWidth + gBoxHeight);
        gPressure = gPressure / (gBoxPerimeter * TOTAL_TIME);
        V = gBoxWidth * gBoxHeight;   // Here Volume = Area.
        printf("\n Total required time = %.2lf seconds\n", endTime - startTime);
        printf("\nPressure = %.3lf  Volume = %.0lf Temperature = %.3lf\n", gPressure, V, gTemp);
        R = (gPressure * V) / (gNumParts * gTemp);
        printf("\t.............................\n");
        printf("\n\t| MAGIC CONSTANT, R = %.3f |\n", R);
        printf("\t.............................\n");
    }
    MPI_Type_free(&mpiPartType);
    MPI_Finalize();
    return 0;
}

float FloatRand(float fMin, float fMax)
{
    float dr = ((float)rand() / (float)(RAND_MAX));
    return fMin + dr * (fMax - fMin);
}

void CalculateGridLayout(int nProcs, int *gridRows, int *gridCols)
{
    int i;
    for (i = sqrt(nProcs); i > 0; i--)
    {
        if (nProcs % i == 0)
        {
            *gridRows = i;
            *gridCols = nProcs / i;
            return;
        }
    }
}

int IsOutsideLocalBox(pcord_t *p, cord_t box)
{
  if(p->x < box.x0)  return LEFT;
  if(p->x >= box.x1)  return RIGHT;
  if(p->y < box.y0) return DOWN;
  if(p->y >= box.y1) return UP;
  return -1;
}

void AddToSendBuffer(particle_t *sendBudAdd, particle_t *p)
{
    memcpy(sendBudAdd, p, sizeof(particle_t));
}
