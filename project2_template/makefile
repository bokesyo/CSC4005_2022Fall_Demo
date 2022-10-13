compile:
	g++ sequential.cpp -o seq -O2 -std=c++11
	mpic++ mpi.cpp -o mpi -std=c++11
	g++ pthread.cpp -lpthread -o pthread -O2 -std=c++11

compileg:
	g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm sequential.cpp -o seqg -DGUI -O2 -std=c++11
	mpic++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm mpi.cpp -o mpig -DGUI -std=c++11
	g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -lpthread pthread.cpp -o pthreadg -DGUI -O2 -std=c++11

clean:
	rm -f seq mpi pthread seqg mpig pthreadg