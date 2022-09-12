# MPI demo

Contact: bokaixu@link.cuhk.edu.cn

## Compile demo code

```bash
mpic++ mpi_hello.cpp -o mpi_hello
```

## Run demo code

```bash
mpirun -np 4 ./mpi_hello
```

The output should be like

```
hello world! Process 0 of 4 on localhost.localdomain
hello world! Process 1 of 4 on localhost.localdomain
hello world! Process 3 of 4 on localhost.localdomain
hello world! Process 2 of 4 on localhost.localdomain
```

# Appendix

## Install mpich-3.2 on Centos:

If you are using Centos

```bash
yum install mpich-3.2 mpich-3.2-devel -y
```

Then add executable to PATH variable:
```bash
vim /etc/profile
```

add `export PATH=$PATH:/usr/lib64/mpich-3.2/bin/` to the last line.

Then
```bash
source /etc/profile
```

## Install on other linux:

https://mpitutorial.com/tutorials/installing-mpich2/



# Reference

[1] https://mpitutorial.com/tutorials/point-to-point-communication-application-random-walk/



