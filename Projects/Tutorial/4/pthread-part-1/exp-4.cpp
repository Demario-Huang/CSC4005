//
// Created by schrodinger on 10/8/21.
//
#include <pthread.h>
#include <iostream>
#include <thread>

pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t counter_cv = PTHREAD_COND_INITIALIZER;
int counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *task(void * data) {
    auto time = reinterpret_cast<size_t>(data);
    std::this_thread::sleep_for(std::chrono::milliseconds{time});

    pthread_mutex_lock(&counter_mutex);
    counter++;
    pthread_mutex_unlock(&counter_mutex);

    pthread_cond_broadcast(&counter_cv);

    {
        pthread_mutex_lock(&counter_mutex);
        while (counter != 6) {
            pthread_cond_wait(&counter_cv, &counter_mutex);
        }
        pthread_mutex_unlock(&counter_mutex);
    }

    pthread_mutex_lock(&mutex);
    std::cout << counter << std::endl;
    pthread_mutex_unlock(&mutex);
    return nullptr;
}

int main() {
    pthread_t threads[6];
    size_t cnt = 0;
    for (auto &i : threads) {
        pthread_create(&i, nullptr, task, reinterpret_cast<void *>(cnt));
        cnt += 10;
    }
    for (auto &i : threads) {
        pthread_join(i, nullptr);
    }
}