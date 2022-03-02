//
// Created by schrodinger on 10/7/21.
//
#include <pthread.h>
#include <iostream>
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int counter_protected = 0;
int counter_relaxed = 0;

void *task_protected(void *) {
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < 100; ++i) {
        counter_protected += i;
        std::cout << i << " ";
    }
    std::cout << std::endl;
    pthread_mutex_unlock(&mutex);
    return nullptr;
}

void *task_relaxed(void *) {
    for (int i = 0; i < 100; ++i) {
        counter_relaxed += i;
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return nullptr;
}

int main() {
    pthread_t threads[4];
    for (auto &i : threads) {
        pthread_create(&i, nullptr, task_protected, nullptr);
    }
    for (auto &i : threads) {
        pthread_join(i, nullptr);
    }
    std::cout << std::endl;
    for (auto &i : threads) {
        pthread_create(&i, nullptr, task_relaxed, nullptr);
    }
    for (auto &i : threads) {
        pthread_join(i, nullptr);
    }
    std::cout << counter_protected << std::endl;
    std::cout << counter_relaxed << std::endl;
}