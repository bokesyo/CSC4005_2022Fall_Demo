#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc,char** argv)
{
    int myid,numproces;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numproces);
    MPI_Get_processor_name(processor_name,&namelen);
    fprintf(stdout,"hello world! Process %d of %d on %s\n",
            myid,numproces,processor_name);
    MPI_Finalize();
 
    return 0;
}

