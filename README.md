# CSC4005 2022Fall Demo Repo

This repo prepares some code templates and demos for CSC4005 programming project. Hope that you can find it useful. These codes can run on both our released VM (CentOS 7) and our HPC cluster (CentOS 7).


## First use

```bash
cd [Your working directory]

git clone https://github.com/bokesyo/CSC4005_2022Fall_Demo.git # clone all the files

chmod -R 777 CSC4005_2022Fall_Demo # make downloaded files executable

```

## Run program

```bash
cd [Your working directory]/CSC4005_2022Fall_Demo/mpi_demo

mpirun -np 4 ./mpi_hello # start 4 process
```

output should be like:

```

hello world! Process 1 of 4 on localhost.localdomain
hello world! Process 3 of 4 on localhost.localdomain
hello world! Process 2 of 4 on localhost.localdomain
hello world! Process 0 of 4 on localhost.localdomain

```

## Update

We may update this repo for bug fix and new demo release, so you can use the following command line to get the latest version.

```bash
cd [Your working directory]/CSC4005_2022Fall_Demo
git fetch # check if there is an update version
git stash # if you have your own change, you have to use this line to store your change first
git pull
```

## Remarks

`template.sh` is the template `sbatch` shell script to run programs on the server. You can adjust the path, nodes, processes, and other parameters in it. Try and play around with it.

## Contribute

Feel free to open an issue for bugs, suggestions, ...

You are welcomed to create your own branch and upload it to Github, so we can improve this repo. 


Maintainer: 

yangzhixinluo@link.cuhk.edu.cn

bokaixu@link.cuhk.edu.cn
