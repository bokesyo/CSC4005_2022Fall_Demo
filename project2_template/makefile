sequential: sequential.cpp ./asg2.h
	g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm sequential.cpp -o sequential

mpi: mpi.cpp ./asg2.h
	mpic++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm mpi.cpp -o mpi

pthread: pthread.cpp ./asg2.h
	g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -lpthread pthread.cpp -o pthread

clean:
	rm -rf sequential mpi pthread
