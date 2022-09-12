# MPI GUI demo

Contact: bokaixu@link.cuhk.edu.cn

## Compile MPI GUI

```bash
mpic++ mpi_gui.cpp -o mpi_gui -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -std=c++11 
```

## Run 

You need to run it in MobaXterm on Windows / XQuartz on mac with X11 forwarding on / terminal on linux with X11 forwarding on (ssh -Y), otherwise GUI will not appear.

```bash
mpirun -np 4 ./mpi_gui
```

# Appendix

mpic++ is a wrapped g++ command, to see how it works, you can try:

```bash
mpic++ -show
```

The output should be like

```bash
g++ -m64 -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=generic -fPIC -Wl,-z,noexecstack -I/usr/include/mpich-3.2-x86_64 -L/usr/lib64/mpich-3.2/lib -lmpicxx -Wl,-rpath -Wl,/usr/lib64/mpich-3.2/lib -Wl,--enable-new-dtags -lmpi
```

which means it is possible to compile an executable with both mpi and opengl, just try:

```bash
g++ -m64 -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=generic -fPIC -Wl,-z,noexecstack -I/usr/include/mpich-3.2-x86_64 -L/usr/lib64/mpich-3.2/lib -lmpicxx -Wl,-rpath -Wl,/usr/lib64/mpich-3.2/lib -Wl,--enable-new-dtags -lmpi mpi_gui.cpp -o mpi_gui -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -std=c++11
```

which has the same result as

```bash
mpic++ mpi_gui.cpp -o mpi_gui -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -std=c++11 
```