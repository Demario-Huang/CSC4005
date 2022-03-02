#include <chrono>
#include <iostream>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <vector>
#include <complex>
#include <mpi.h>
#include <cstring>

struct Square {
    std::vector<int> buffer;
    size_t length;

    explicit Square(size_t length) : buffer(length), length(length * length) {}

    void resize(size_t new_length) {
        buffer.assign(new_length * new_length, false);
        length = new_length;
    }

    auto& operator[](std::pair<size_t, size_t> pos) {
        return buffer[pos.second * length + pos.first];
    }
};

/* declare the global variables*/
#define MAX_ITERATIONS 100
int center_x = 0;
int center_y = 0;
int size = 800;
int scale = 1;
int k_value = 100;

/* used for draw the picture */
Square canvas(size);
static constexpr float MARGIN = 4.0f;
static constexpr float BASE_SPACING = 2000.0f;
static constexpr size_t SHOW_THRESHOLD = 500000000ULL;

int main(int argc, char **argv) {
    int rank;
    int mpisize;
    int res;
    MPI_Init(&argc, &argv);
    res = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    res = MPI_Comm_size(MPI_COMM_WORLD,&mpisize);

    if (MPI_SUCCESS != res) {
        throw std::runtime_error("failed to get MPI world rank");
    }

    /* boardcast the variable to each rank */
    MPI_Bcast(&center_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scale, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0){

        MPI_Request request;
        MPI_Status status;     


        while (true){
            
            /* update the variables for each rank */
            MPI_Bcast(&center_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&center_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&k_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&scale, 1, MPI_INT, 0, MPI_COMM_WORLD);

            /* calculate the length of tasks for each rank*/
            int local_len =  size / mpisize;
            int remain = size % mpisize;
            if (rank < remain) local_len++;

            int * localarr = (int*)std::malloc(sizeof(int) * size * local_len); 

            /* Recieve the value from root */     
            MPI_Irecv(localarr, size*local_len ,MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
          
            double cx = static_cast<double>(size) / 2 + center_x;
            double cy = static_cast<double>(size) / 2 + center_y;
            double zoom_factor = static_cast<double>(size) / 4 * scale;

            /* calcuate the k and store in the local array */
            int min;
            int max; 
            if (rank < remain) min = rank*local_len;
            else min = remain * (local_len + 1) + (rank -  remain) * (local_len);  
            max = min + local_len;

            for (int i = min; i < max; ++i){
                for (int j = 0; j < size; ++j){
                    double x = (static_cast<double>(j) - cx) / zoom_factor;
                    double y = (static_cast<double>(i) - cy) / zoom_factor;
                    std::complex<double> z{0, 0};
                    std::complex<double> c{x, y};
                    int k = 0;
                    do {
                        z = z * z + c;
                        k++;
                    } while (norm(z) < 2.0 && k < k_value);
                    localarr[j + (i - min)*size ] = k;
                }
            }

            /* send the calculation result from local array to global array*/
            MPI_Request sendreq; 
            MPI_Isend(localarr, local_len * size, MPI_INT, 0, 0, MPI_COMM_WORLD, &sendreq);
            MPI_Wait(&sendreq, MPI_STATUSES_IGNORE);
            std::free(localarr);
        }
    }

    if (0 == rank) {
        graphic::GraphicContext context{"Assignment 2"};
        size_t duration = 0;
        size_t pixels = 0;
        // int count = 0;
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
            {   
                auto io = ImGui::GetIO();
                ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
                ImGui::SetNextWindowSize(io.DisplaySize);
                ImGui::Begin("Assignment 2", nullptr,
                             ImGuiWindowFlags_NoMove
                             | ImGuiWindowFlags_NoCollapse
                             | ImGuiWindowFlags_NoTitleBar
                             | ImGuiWindowFlags_NoResize);
                ImDrawList *draw_list = ImGui::GetWindowDrawList();
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);
                static ImVec4 col = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
                ImGui::DragInt("Center X", &center_x, 1, -4 * size, 4 * size, "%d");
                ImGui::DragInt("Center Y", &center_y, 1, -4 * size, 4 * size, "%d");
                ImGui::DragInt("Fineness", &size, 10, 100, 1000, "%d");
                ImGui::DragInt("Scale", &scale, 1, 1, 100, "%.01f");
                ImGui::DragInt("K", &k_value, 1, 100, 1000, "%d");
                ImGui::ColorEdit4("Color", &col.x);
                {
                    using namespace std::chrono;
                    auto spacing = BASE_SPACING / static_cast<float>(size);
                    auto radius = spacing / 2;
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    const ImU32 col32 = ImColor(col);
                    float x = p.x + MARGIN, y = p.y + MARGIN;
                    canvas.resize(size);
                    auto begin = high_resolution_clock::now();

                    ////////////// CODE HERE ////////////
                    {   

                        /* every time update the loop and boardcast the variables */
                        MPI_Bcast(&center_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        MPI_Bcast(&center_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        MPI_Bcast(&k_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        MPI_Bcast(&scale, 1, MPI_INT, 0, MPI_COMM_WORLD);

                        /* create a global array to store the k */
                        int * globalarr = (int*)malloc( sizeof(int) * size * size);

                        /* send the array to another rank to calculate */
                        for (int i = 1; i < mpisize; i++){

                            int local_len =  size / mpisize;
                            int remain = size % mpisize;
                            if (i < remain) local_len++;
                            int * localarr = (int*)std::malloc(sizeof(int) * size * local_len);

                            MPI_Request request;
                            MPI_Isend(localarr, local_len * size, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
                            MPI_Wait(&request, MPI_STATUSES_IGNORE);
                            free(localarr);                   
                        }

                        /* calcualte the root  array */
                        int root_len = size / mpisize;
                        int remain = size % mpisize;
                        if ( remain != 0) root_len++;
                        int * rootarr = (int *)malloc(sizeof(int) * root_len * size);
                        double cx = static_cast<double>(size) / 2 + center_x;
                        double cy = static_cast<double>(size) / 2 + center_y;
                        double zoom_factor = static_cast<double>(size) / 4 * scale;
                        for (int i = 0; i < root_len; ++i){
                            for (int j = 0; j < size; ++j){
                                double x = (static_cast<double>(j) - cx) / zoom_factor;
                                double y = (static_cast<double>(i) - cy) / zoom_factor;
                                std::complex<double> z{0, 0};
                                std::complex<double> c{x, y};
                                int k = 0;
                                do {
                                    z = z * z + c;
                                    k++;
                                } while (norm(z) < 2.0 && k < k_value);
                                rootarr[j + size * i] = k;
                            }
                        }

                        /* send the root calculation result to global array  */
                        for ( int i = 0; i < root_len*size; i++){
                            globalarr[i] = rootarr[i];
                        }
                        free(rootarr);

                        /* Recieve the result from the other rank and store the value in array */
                        for (int i = 1; i < mpisize; ++i){
                            int local_len =  size / mpisize;
                            int remain = size % mpisize;
                            if (i < remain) local_len++;
                            MPI_Request request;
                            MPI_Status status;     
                            int * localarr = (int*)std::malloc(sizeof(int) * size * local_len);
                            MPI_Irecv(localarr, size*local_len ,MPI_INT, i, 0, MPI_COMM_WORLD, &request);
                            MPI_Wait(&request, &status);
                            int index ;
                            if ( i < remain) index = local_len*size*i;
                            else index = (local_len+1)*remain*size + (i-remain)*local_len*size;
                            for (int j = 0; j < size*local_len; j++){
                                globalarr[index + j] = localarr[j];
                            }
                            free(localarr);
                        }

                        /* write the each array value into canvas */
                        for (int i = 0; i < size; ++i){
                            for (int j = 0; j < size; ++j) {
                                canvas[{i,j}] = globalarr[j + i*size];
                            }
                        }
                        free(globalarr);   

                    }
                    ////////////// CODE END /////////////

                    auto end = high_resolution_clock::now();
                    pixels += size;
                    duration += duration_cast<nanoseconds>(end - begin).count();
                    if (duration > SHOW_THRESHOLD ) {
                        std::cout << pixels << " pixels in last " << duration << " nanoseconds\n";
                        auto speed = static_cast<double>(pixels) / static_cast<double>(duration) * 1e9;
                        std::cout << "speed: " << speed << " pixels per second" << std::endl;
                        pixels = 0;
                        duration = 0;
                    }
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            if (canvas[{i, j}] == k_value) {
                                draw_list->AddCircleFilled(ImVec2(x, y), radius, col32);
                            }
                            x += spacing;
                        }
                        y += spacing;
                        x = p.x + MARGIN;
                    }
                }
                ImGui::End();
            }
        });
    }
    MPI_Finalize();
    return 0;
}
