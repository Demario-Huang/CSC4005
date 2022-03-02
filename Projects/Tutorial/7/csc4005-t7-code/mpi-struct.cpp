#include <cstdio>
#include <mpi.h>

using namespace std;

struct Square {
  int height;
  int width;
  int price;
} __attribute__((packed));

void describeSquare(int rank, const Square sq) {
  printf("[Rank %d] Square[%dx%d, $%d]\n", rank, sq.height, sq.width, sq.price);
}

int main() {
  int rank, size;

  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Declare a datatype
  MPI_Datatype MPISquare;
  MPI_Type_contiguous(3, MPI_INT, &MPISquare);
  MPI_Type_commit(&MPISquare);

  if (rank == 0) {
    // create an Square
    Square sqsend = {123, 346, 765};
    describeSquare(0, sqsend);
    MPI_Send(&sqsend, 1, MPISquare, 1, 0, MPI_COMM_WORLD);
  } else {
    Square sqrecv = {1, 1, 1};
    describeSquare(1, sqrecv);
    MPI_Recv(&sqrecv, 1, MPISquare, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    describeSquare(1, sqrecv);
  }

  MPI_Finalize();
  return 0;
}
