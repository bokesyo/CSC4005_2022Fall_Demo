compile: 
nvcc cuda.cu -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm --std=c++11
run:
./a.out 10000 100000