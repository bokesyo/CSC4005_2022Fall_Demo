compile:
	g++ ./src/sequential.cpp -o seq -O2 -std=c++11
	mpic++ ./src/mpi.cpp -o mpi -std=c++11
	g++ ./src/pthread.cpp -lpthread -o pthread -O2 -std=c++11

compileg:
	g++ ./src/sequential.cpp -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -o seqg -DGUI -O2 -std=c++11
	mpic++ ./src/mpi.cpp -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -o mpig -DGUI -std=c++11
	g++ ./src/pthread.cpp -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -lpthread -o pthreadg -DGUI -O2 -std=c++11

clean:
	rm -f seq mpi pthread seqg mpig pthreadg