#include <iostream>
#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#define MOD 1000000007
struct Arguments {
    int length;
    int *data;
    std::atomic<int> *output;
};

void *task(void *arg_ptr) {
    auto arguments = static_cast<Arguments *>(arg_ptr);
    int acc = 0;
    for (int i = 0; i < arguments->length; ++i) {
        acc += arguments->data[i];
        acc %= MOD;
    }
    *arguments->output += acc;
    delete arguments;
    return nullptr;
}

int get_length(int num, int thd, int idx) {
    return ((num - idx) % thd > 0) + (num - idx) / thd;
}

int main() {
    using namespace std::chrono;
    std::random_device dev;
    std::default_random_engine eng(dev());
    std::uniform_int_distribution<int> dist {0};
    int thread_num, data_num;
    std::cout << "thread number: ";
    std::cout.flush();
    std::cin >> thread_num;
    std::cout << "data number: ";
    std::cout.flush();
    std::cin >> data_num;
    if (thread_num > data_num) {
        thread_num = data_num;
    }
    std::vector<int> data(data_num);
    std::cout << "hello world " << std::endl;
    std::vector<pthread_t> threads(thread_num);
    std::atomic<int> answer{0};
    for (auto &i : data) { i = dist(eng); }
    auto start = high_resolution_clock::now();
    int count = 0;
    for (int i = 0; i < threads.size(); ++i) {
        auto length = get_length(data_num, thread_num, i);
        pthread_create(&threads[i], nullptr, task, new Arguments{
                .length = length,
                .data = data.data() + count,
                .output = &answer
        });
        count += length;
    }
    for (auto & i : threads) {
        pthread_join(i, nullptr);
    }
    auto end = high_resolution_clock::now();
    std::cout << answer.load() % MOD << std::endl;
    std::cout << "ns: " <<duration_cast<nanoseconds>(end - start).count() << std::endl;
}
