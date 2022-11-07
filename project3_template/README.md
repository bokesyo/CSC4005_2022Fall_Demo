# CSC4005 Project 3 Template

## Physics

We have some physics variable declared in `headers/physics.h`:

```c++
#define gravity_const 1.0f
#define dt 0.0001f
#define error 1e-9f
#define radius2 4.0f
#define bound_x 4000
#define bound_y 4000
#define max_mass 40000000
```

`gravity_const` is the gravity constant when you compute $F=G m_{i} m_{j} / d^2$.

`dt` is the time span between two iterations, it can be used when you compute $\Delta v=F\Delta t$ and $\Delta x=v\Delta t$.

`error` is a small number used to avoid `DivisionByZero` error. It can be used like $F=G m_{i} m_{j} / (d^2 + error)$.

`radius2` is the squared radius of particles. It can be used when you determine whether two particles have collision.

`bound_x` is the upper bound of X axis. $x$ is between $[0,bound\_x]$.

`bound_y` is the upper bound of Y axis. $y$ is between $[0,bound\_y]$.

`max_mass` is the maximum mass of a particle. You can use it to generate particles.

You may need to modify them to better visualize your result.


## Compile

Sequential (command line application):

```bash
g++ ./src/sequential.cpp -o seq -O2 -std=c++11
```

Sequential (GUI application):

```bash
g++ ./src/sequential.cpp -o seqg -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -DGUI -O2 -std=c++11
```

MPI (command line application):

```bash
mpic++ ./src/mpi.cpp -o mpi -std=c++11
```

MPI (GUI application):

```bash
mpic++ ./src/mpi.cpp -o mpig -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -DGUI -std=c++11
```

Pthread (command line application):

```bash
g++ ./src/pthread.cpp -o pthread -lpthread -O2 -std=c++11
```

Pthread (GUI application):

```bash
g++ ./src/pthread.cpp -o pthreadg -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -lpthread -DGUI -O2 -std=c++11
```

CUDA (command line application): notice that `nvcc` is not available on VM, please use cluster.

```bash
nvcc ./src/cuda.cu -o cuda -O2 --std=c++11
```

CUDA (GUI application): notice that `nvcc` is not available on VM, please use cluster.

```bash
nvcc ./src/cuda.cu -o cudag -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -O2 -DGUI --std=c++11
```


OpenMP (command line application):

```bash
g++ ./src/openmp.cpp -o openmp -fopenmp -O2 -std=c++11
```

OpenMP (GUI application):

```bash
g++ ./src/openmp.cpp -o openmpg -fopenmp -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -O2 -DGUI -std=c++11
```


## Run

Sequential (command line mode):

```bash
./seq $n_body $n_iterations
```

Sequential (GUI mode): please run this on VM (with GUI desktop).

```bash
./seqg $n_body $n_iterations
```

MPI (command line mode):

```bash
mpirun -np $n_processes ./mpi $n_body $n_iterations
```

MPI (GUI mode): please run this on VM (with GUI desktop).

```bash
mpirun -np $n_processes ./mpig $n_body $n_iterations
```


Pthread (command line mode):

```bash
./pthread $n_body $n_iterations $n_threads
```

Pthread (GUI mode): please run this on VM (with GUI desktop).

```bash
./pthreadg $n_body $n_iterations $n_threads
```

CUDA (command line mode): for VM users, please run this on cluster.

```bash
./cuda $n_body $n_iterations
```

CUDA (GUI mode): if you have both nvcc and GUI desktop, you can try this.

```bash
./cuda $n_body $n_iterations
```


OpenMP (command line mode):

```bash
openmp $n_body $n_iterations $n_omp_threads
```

OpenMP (GUI mode):

```bash
openmpg $n_body $n_iterations $n_omp_threads
```



## Makefile

Makefile helps you simplify compilation command.

```bash
make $command
```

where `command` is one of `seq, seqg, mpi, mpig, pthread, pthreadg, cuda, cudag`.