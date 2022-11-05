# CSC4005 Project 3 Template

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

CUDA (command line application): notice that `nvcc` is not available on VM.

```bash
nvcc ./src/cuda.cu -o cuda -O2 --std=c++11
```

CUDA (GUI application):

```bash
nvcc ./src/cuda.cu -o cudag -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -O2 -DGUI --std=c++11
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


## Makefile

Makefile helps you simplify compilation command.

```bash
make $command
```

where `command` is one of `seq, seqg, mpi, mpig, pthread, pthreadg, cuda, cudag`.