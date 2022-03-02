#pragma once

#include <vector>
#include <mpi.h>
#include <omp.h>
#include <iostream>

namespace hdist {

    enum class Algorithm : int {
        Jacobi = 0,
        Sor = 1
    };

    struct State {
        int room_size = 300;
        float block_size = 2;
        int source_x = room_size / 2;
        int source_y = room_size / 2;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 4.0;
        Algorithm algo = hdist::Algorithm::Jacobi;

        bool operator==(const State &that) const = default;
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

    bool calculate(const State &state, Grid &grid, int rank, int proc_num, int threadnum) {
        
        bool stabilized = true;

        int mylen = state.room_size  / proc_num;
        int remain = state.room_size % proc_num;
        if (rank < remain) mylen++;

        size_t min;
        size_t max;

        if (rank < remain) min = mylen * rank;
        else min = remain * (mylen+1) + (rank - remain) * mylen;
        max = min + mylen;

        int * recvcounts = (int *)malloc(sizeof(int) * proc_num);
        int * displ = (int *)malloc(sizeof(int) * proc_num);

        for (int rankid = 0; rankid < proc_num  ; rankid++){
            int curlen = state.room_size / proc_num;
            int curremain = state.room_size  % proc_num;
            if (rankid < curremain) curlen++;
            recvcounts[rankid] = curlen * state.room_size;

            int curmin;
            if(rankid < curremain) curmin = curlen * rankid * state.room_size;
            else curmin = curremain * (curlen + 1) * state.room_size + (rankid - curremain)* curlen * state.room_size;
            displ[rankid] = curmin;
        }


        switch (state.algo) {
            case Algorithm::Jacobi:
                for (size_t i = min; i < max; ++i) {
                    #pragma omp parallel num_threads(threadnum)
                    {
                        int ompid = omp_get_thread_num();
                        int ompsize = state.room_size / threadnum;
                        int ompremain = state.room_size % threadnum;
                        if(ompid < ompremain) ompsize++;

                        size_t ompmin;
                        size_t ompmax;

                        if(ompid < ompremain ) ompmin = ompsize * ompid;
                        else ompmin = ompremain * (ompsize + 1) + (ompid - ompremain) * ompsize;
                        ompmax = ompmin + ompsize;

                        for (size_t j = ompmin; j < ompmax; ++j) {
                            auto result = update_single(i, j, grid, state);
                            stabilized &= result.stable;
                            grid[{alt, i, j}] = result.temp;
                        }
                    }
                }
                grid.switch_buffer();
                MPI_Allgatherv(&grid.get_current_buffer()[displ[rank]], recvcounts[rank] , MPI_DOUBLE, &grid.get_current_buffer()[0],recvcounts,displ,MPI_DOUBLE,MPI_COMM_WORLD );
                break;
            case Algorithm::Sor:
                for (auto k : {0, 1}) {
                    for (size_t i = min; i < max; i++) {
                        #pragma omp parallel num_threads(threadnum)
                        {
                            int ompid = omp_get_thread_num();
                            int ompsize = state.room_size / threadnum;
                            int ompremain = state.room_size % threadnum;
                            if(ompid < ompremain) ompsize++;

                            size_t ompmin;
                            size_t ompmax;

                            if(ompid < ompremain ) ompmin = ompsize * ompid;
                            else ompmin = ompremain * (ompsize + 1) + (ompid - ompremain) * ompsize;
                            ompmax = ompmin + ompsize;

                            for (size_t j = ompmin; j < ompmax; j++) {
                                if (k == ((i + j) & 1)) {
                                    auto result = update_single(i, j, grid, state);
                                    stabilized &= result.stable;
                                    grid[{alt, i, j}] = result.temp;
                                } else {
                                    grid[{alt, i, j}] = grid[{i, j}];
                                }
                            }
                        }
                    }
                    grid.switch_buffer();
                    MPI_Allgatherv(&grid.get_current_buffer()[displ[rank]], recvcounts[rank] , MPI_DOUBLE, &grid.get_current_buffer()[0],recvcounts,displ,MPI_DOUBLE,MPI_COMM_WORLD );
                    MPI_Bcast(&grid.get_current_buffer()[0], state.room_size * state.room_size , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                }
        }


        return stabilized;
    };


} // namespace hdist
