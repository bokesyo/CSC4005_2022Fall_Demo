#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>

using namespace std;

void *welcome(void *arg) {
    cout << "Id: " << pthread_self() << endl;
    cout << "Welcome to Pthreads Programming" << endl;
    return (void *)0;
}

int main() {
    int ret;
    int *stat;
    pthread_t tid;

    // Create a thread within the process to execute welcome

    if ((ret = pthread_create(&tid, NULL, welcome, NULL)) != 0) {
        cout << "Error creating thread: " << strerror(ret) << endl;
        exit(1);
    }

    cout << "Created Thread " << tid << endl;
    pthread_join(tid, (void **)&stat);
    cout << "Thread " << tid << " terminated, Status = " << stat << endl;
    exit(0);

}
