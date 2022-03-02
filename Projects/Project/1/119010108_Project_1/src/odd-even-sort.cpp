#include <odd-even-sort.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <stdlib.h>

namespace sort {

    using namespace std::chrono;

    Context::Context(int &argc, char **&argv) : argc(argc), argv(argv) {
        MPI_Init(&argc, &argv);
    }

    Context::~Context() {
        MPI_Finalize();
    }

    std::unique_ptr<Information> Context::mpi_sort(Element *begin, Element *end) const {
        int res;
        int rank;
        std::unique_ptr<Information> information{};

        res = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (MPI_SUCCESS != res) {
            throw std::runtime_error("failed to get MPI world rank");
        }


        if (0 == rank) {
            information = std::make_unique<Information>();
            information->length = end - begin;
            res = MPI_Comm_size(MPI_COMM_WORLD, &information->num_of_proc);
            if (MPI_SUCCESS != res) {
                throw std::runtime_error("failed to get MPI world size");
            };
            information->argc = argc;
            for (auto i = 0; i < argc; ++i) {
                information->argv.push_back(argv[i]);
            }
            information->start = high_resolution_clock::now();
        }

        /* boardcast the array size and number of processors to each mpi program */
        size_t num = 1;
        int arrsize;
        int numofproc;
        if (0 == rank){
            arrsize = information->length;
            numofproc = information->num_of_proc;
        }
        MPI_Bcast(&arrsize,num,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(&numofproc,num,MPI_INT,0,MPI_COMM_WORLD);

        
        /* malloc the new array to store original array then boardcast to each mpi program */
        int64_t * global_arr = (int64_t*)malloc(sizeof(int64_t) * arrsize);
        if (0 == rank){
            for (int i = 0; i < arrsize; i++){
                global_arr[i] = *(begin + i);
            }
        }
        MPI_Bcast(global_arr,arrsize,MPI_INT,0,MPI_COMM_WORLD);
        

        int local_len;
        int *sendcounts = (int*)malloc(sizeof(int) * numofproc);
        int *displs = (int*)malloc(sizeof(int)* numofproc);

        /* used for calculating the number of processors for each mpi program */
        int reminder = arrsize % numofproc;
        int sum = 0;
        for (int i = 0; i < numofproc; i++) {
            sendcounts[i] = arrsize/numofproc;
            if (reminder > 0) {
                sendcounts[i]++;
                reminder--;
            }
            displs[i] = sum;
            sum += sendcounts[i];
        }   

        /* distribute the processors for each mpi program */
        for (int count = 0; count < numofproc; count++) {
            if (rank == count) {
                local_len = sendcounts[count];
                std::cout << "The rank " << rank << " have: " << local_len << " processors. " << std::endl;
            }
        }
        int64_t* local_array = (int64_t*)malloc(sizeof(int64_t) *local_len);

        /* Scatter the array information from root to another mpi program */
        MPI_Scatterv(global_arr,sendcounts,displs,MPI_LONG,local_array,local_len,MPI_LONG,0, MPI_COMM_WORLD);


        /* steps for Parallel Odd-Even Transposition Sort*/
        {
            int64_t send_val = 0;
            int64_t recv_val = 1;
            int rightrank = (rank +  1) % (numofproc);
            int leftrank = (rank + numofproc - 1) % (numofproc);

            for( int count = 0; count < arrsize; count++ ) {
                if (count%2 == 0) {
                    for (int k = local_len - 1 ; k > 0; k-=2){
                        if (local_array[k] < local_array[k-1]){
                            std::swap(local_array[k], local_array[k - 1]);
                        }
                    }
                }

                else{
                    for (int j = local_len - 2; j > 0; j -= 2){
                        if (local_array[j] < local_array[j - 1]){
                            std::swap(local_array[j], local_array[j - 1]);
                        }
                    }

                    if (rank != 0){
                        send_val = local_array[0];
                        MPI_Send(&send_val, 1, MPI_LONG, leftrank, 0, MPI_COMM_WORLD);
                        MPI_Recv(&recv_val, 1, MPI_LONG, leftrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (recv_val > local_array[0]) local_array[0] = recv_val;
                    }

                    if (rank != numofproc - 1){
                        int64_t send_newval = local_array[local_len - 1];
                        MPI_Recv(&recv_val, 1, MPI_LONG, rightrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                        MPI_Send(&send_newval, 1, MPI_LONG, rightrank, 0, MPI_COMM_WORLD);
                        if (recv_val < local_array[local_len - 1]) local_array[local_len - 1] = recv_val;
                    }
                }
            }

        }

        MPI_Allgatherv(local_array, local_len,MPI_LONG, global_arr,sendcounts,displs,MPI_LONG, MPI_COMM_WORLD);

        /* pass the ordered array to the original one */
        if (0 == rank){
            for (int i = 0; i < arrsize; i++){
                *(begin+i) = global_arr[i];
            }

            std::cout << "the Sorted array is: "; 
            for (int i =0; i < arrsize; i++){
                if (i == arrsize -1){
                    std::cout << *(begin + i) << std::endl;
                }
                else{
                    std::cout << *(begin + i) << "->";
                }

            }

        }

        if (0 == rank) {
            information->end = high_resolution_clock::now();
        }

        free(global_arr);
        free(local_array);
        free(sendcounts);
        free(displs);

        return information;


    }

    std::ostream &Context::print_information(const Information &info, std::ostream &output) {
        auto duration = info.end - info.start;
        auto duration_count = duration_cast<nanoseconds>(duration).count();
        auto mem_size = static_cast<double>(info.length) * sizeof(Element) / 1024.0 / 1024.0 / 1024.0;
        output << "input size: " << info.length << std::endl;
        output << "proc number: " << info.num_of_proc << std::endl;
        output << "duration (ns): " << duration_count << std::endl;
        output << "throughput (gb/s): " << mem_size / static_cast<double>(duration_count) * 1'000'000'000.0
               << std::endl;
        return output;
    }
}
