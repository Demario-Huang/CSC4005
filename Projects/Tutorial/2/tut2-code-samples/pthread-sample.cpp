// clang++ -pthread pthread-sample.cpp
#include <cstdio>
#include <vector>
#include <thread>

int main() {
    using namespace std;
    vector<thread> pool{};
    for(int i=0; i<5; i++) {
        pool.emplace_back([i] {
            printf("hello world from #%d\n", i);
        });
    }
    for(int i=0; i<5; i++) {
        if (pool[i].joinable()) {
            pool[i].join();
        }
    }
    return 0;
}
