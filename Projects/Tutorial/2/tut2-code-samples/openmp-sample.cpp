// clang++ -fopenmp openmp-sample.cpp
#include <cstdio>
#include <omp.h>

int main(int argc, char *argv[]) {
    using namespace std;
    omp_set_num_threads(4);
    
    #pragma omp parallel for 
    for (int i=1; i<20; i++)
        printf("hello world from #%d\n", i);
    return 0;
}
