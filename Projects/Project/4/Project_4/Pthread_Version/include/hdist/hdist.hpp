#pragma once

#include <vector>
#include <pthread.h>
#include <iostream>
#include <unistd.h>

#define MAX_THREADS 128



namespace hdist {

    void * Pthread_calculate(void * t);

    enum class Algorithm : int {
        Jacobi = 0,
        Sor = 1
    };

    struct State {
        int room_size = 300;
        float block_size = 2;
        int source_x = room_size / 2 ;
        int source_y = room_size / 2 ;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 4.0;
        Algorithm algo = hdist::Algorithm::Jacobi;

        //bool operator==(const State &that) const = default;
    };

    struct Alt {
    };

    constexpr static inline Alt alt{};

    struct Grid {
        std::vector<double> data0, data1;
        size_t current_buffer = 0;
        size_t length;

        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
                : data0(size * size), data1(size * size), length(size) {
            for (size_t i = 0; i < length; ++i) {
                for (size_t j = 0; j < length; ++j) {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1) {
                        this->operator[]({i, j}) = border_temp;
                    } else if (i == x && j == y) {
                        this->operator[]({i, j}) = source_temp;
                    } else {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }

        std::vector<double> &get_current_buffer() {
            if (current_buffer == 0) return data0;
            return data1;
        }

        double &operator[](std::pair<size_t, size_t> index) {
            return get_current_buffer()[index.first * length + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index) {
            return current_buffer == 1 ? data0[std::get<1>(index) * length + std::get<2>(index)] : data1[
                    std::get<1>(index) * length + std::get<2>(index)];
        }

        void switch_buffer() {
            current_buffer = !current_buffer;
        }
    };

    struct UpdateResult {
        bool stable;
        double temp;
    };

    UpdateResult update_single(size_t i, size_t j, Grid &grid, const State &state) {
        UpdateResult result{};
        if (i == 0 || j == 0 || i == state.room_size - 1 || j == state.room_size - 1) {
            result.temp = state.border_temp;
        } else if (i == state.source_x && j == state.source_y) {
            result.temp = state.source_temp;
        } else {
            auto sum = (grid[{i + 1, j}] + grid[{i - 1, j}] + grid[{i, j + 1}] + grid[{i, j - 1}]);
            switch (state.algo) {
                case Algorithm::Jacobi:
                    result.temp = 0.25 * sum;
                    break;
                case Algorithm::Sor:
                    result.temp = grid[{i, j}] + (1.0 / state.sor_constant) * (sum - 4.0 * grid[{i, j}]);
                    break;
            }
        }
        result.stable = fabs(grid[{i, j}] - result.temp) < state.tolerance;
        return result;
    }

    /* used to pass the variable for each thread */
    struct threadpara
    {
        int thread_id;
        int thread_num;
        State * state;
        Grid  * grid;
        bool * stabilized;
    };

    /* deifine the pthread variable */
    pthread_attr_t attr;
    threadpara thread_paras[MAX_THREADS];

    bool calculate(State &state, Grid &grid, int threadnum) {
        bool stabilized = true;

        /* create the thread */
        pthread_t * threads = (pthread_t *) malloc(sizeof(pthread_t)* threadnum);

        /* create the local stable to avoid the data race */
        bool      * each_stable = (bool *)  malloc(sizeof(bool) * threadnum);

        for (int j = 0; j < threadnum; ++j){
            each_stable[j] = stabilized;
        }

        /* used to pass the information to threads */
        for (int i = 0; i < threadnum; i++) {
            thread_paras[i].thread_id = i;
            thread_paras[i].thread_num = threadnum;
            thread_paras[i].state = &state;
            thread_paras[i].grid =  &grid;
            thread_paras[i].stabilized = &each_stable[i];
        }

        /* initial the thread information */
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

        /* create the threads */
        for (int j = 0; j < threadnum; j++) {
            pthread_create(&threads[j],&attr,Pthread_calculate,(void*)&thread_paras[j]);
        }

        /* join the all the threads*/
        for (int i = 0; i < threadnum; i++){
            pthread_join(threads[i],NULL);
        }
    
        pthread_attr_destroy(&attr);
        free(threads);

        /* join each local stable */
        for (int j = 0; j < threadnum; ++j){
            stabilized &= each_stable[j];
        }

        free(each_stable);

        return stabilized;
    };

    void * Pthread_calculate(void * t){
        
        /* pass the variable to each thread, use pointer to pass the grid and state  */
        struct threadpara *recv_para = (struct threadpara*)t;
        int myid = (*recv_para).thread_id;
        int threadnum = (*recv_para).thread_num;
        State *state = (*recv_para).state;
        Grid *grid = (*recv_para).grid;
        bool *stabilized = (*recv_para).stabilized;

        /* calculate the start and end point */
        int mylen = (*state).room_size / threadnum;
        int remain = (*state).room_size % threadnum;
        if (myid < remain) mylen++;

        size_t min;
        size_t max;

        if (myid < remain) min = mylen * myid;
        else min = remain * (mylen+1) + (myid - remain) * mylen;
        max = min + mylen;

        switch((*state).algo){
            case Algorithm::Jacobi:
                for (size_t i = min; i < max; ++i) {
                    for (size_t j = 0; j < (*state).room_size; ++j){
                        auto result = update_single(i, j, (*grid), (*state));
                        (*stabilized) &= result.stable;
                        (*grid)[{alt, i, j}] = result.temp;
                    }
                }
                /* only switch once */
                if(myid == 0) (*grid).switch_buffer();
                break;
            
            case Algorithm::Sor:
                for (auto k : {0, 1}){
                    for (size_t i = min; i < max; ++i){
                        for (size_t j = 0; j < (*state).room_size; j++){
                            if (k == ((i + j) & 1)){
                                auto result = update_single(i, j, (*grid), (*state));
                                (*stabilized) &= result.stable;
                                (*grid)[{alt, i, j}] = result.temp;
                            }else{
                                (*grid)[{alt, i, j}] = (*grid)[{i, j}];
                            }
                        }
                    }
                    (*grid).switch_buffer();
                }
        }

        pthread_exit(NULL);
    };


} // namespace hdist