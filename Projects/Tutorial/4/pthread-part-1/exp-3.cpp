//
// Created by schrodinger on 10/8/21.
//
#include <pthread.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>

pthread_barrier_t barrier;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
std::atomic_int counter{0};

void *task(void *data) {
    auto time = reinterpret_cast<size_t>(data);
    std::this_thread::sleep_for(std::chrono::milliseconds{time});
    counter += 10;
    pthread_barrier_wait(&barrier);
    pthread_mutex_lock(&mutex);
    std::cout << counter << std::endl;
    pthread_mutex_unlock(&mutex);
    return nullptr;
}

int main() {
    pthread_barrier_init(&barrier, nullptr, 6);
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
