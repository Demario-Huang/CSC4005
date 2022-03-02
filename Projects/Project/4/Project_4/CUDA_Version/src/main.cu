#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <stdio.h>

/* used for cuda device to pass data */
struct CU_UpdateResult {
    bool stable;
    double temp;
};

ImColor temp_to_color(double temp) {
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

/* rewrite the calculation for each single grid for cuda to use */
__device__ CU_UpdateResult CU_update_single(size_t i, size_t j, int room_size, int source_x, int source_y ,float source_temp, float border_temp, float tolerance, float sor_constant,
                                            int Algo, int current_buffer, double * data0, double * data1){
    CU_UpdateResult result{};

    if(i == 0 || j == 0 || i == room_size -1 || j == room_size -1){
        result.temp = border_temp;
    }
    else if (i == source_x && j == source_y){
        result.temp = source_temp;
    }
    else{
        double  sum;
        if (current_buffer == 0)  sum = data0[(i+1) * room_size + j] + data0[(i-1)*room_size + j] + data0[i*room_size +j+1] + data0[i*room_size +j-1];
        else  sum = data1[(i+1) * room_size + j] + data1[(i-1)*room_size + j] + data1[i*room_size +j+1] + data1[i*room_size +j-1];
        switch(Algo){
            case 0: // Jacobi
                result.temp = 0.25 * sum;
                break;
            case 1:
                if (current_buffer == 0) result.temp = data0[i*room_size + j] + (1.0 / sor_constant) * (sum - 4.0* data0[i*room_size + j]);
                else result.temp = data1[i*room_size + j] + (1.0 / sor_constant) * (sum - 4.0* data1[i*room_size + j]);
                break;
        }
    }
    if (current_buffer == 0 ) result.stable = fabs(data0[i*room_size + j] - result.temp) < tolerance;
    else result.stable = fabs(data1[i*room_size + j] - result.temp) < tolerance;
    return result;
}

/* kernel function */
__global__ void mykernel(int room_size, int source_x, int source_y, float source_temp, float border_temp, float tolerance, float sor_constant, int Algo, int current_buffer,
                         int threadnum, double * data0, double * data1, int * CUDA_finish ){

    int rank = threadIdx.x;

    bool stabilized = true;

    int mylen = room_size / threadnum;
    int remain = room_size % threadnum;
    if (rank < remain) mylen++;

    size_t min;
    size_t max;

    if (rank < remain) min = mylen * rank;
    else min = remain * (mylen + 1) + (rank - remain) * mylen;
    max= min + mylen;

    switch(Algo){
        case 0: // Jacobi
            for (size_t i = min; i < max; ++i){
                for(size_t j = 0; j < room_size; ++j){
                    auto result = CU_update_single(i, j, room_size, source_x, source_y, source_temp, border_temp, tolerance, sor_constant,  
                                                    Algo, current_buffer, data0, data1 );
                    stabilized &= result.stable;
                    if(current_buffer == 0) data1[i*room_size + j] = result.temp;
                    else data0[i * room_size + j] = result.temp;
                }
            }
            break;
        case 1: // Sor
            for (auto k: {0,1}){
                for (size_t i = min; i < max; ++i){
                    for(size_t j = 0; j < room_size; ++j ){
                        if(k == ((i+j) & 1)){
                            auto result = CU_update_single(i, j, room_size, source_x, source_y, source_temp, border_temp, tolerance, sor_constant,  
                                Algo, current_buffer, data0, data1 );
                            stabilized &= result.stable;
                            if(current_buffer == 0) data1[i*room_size + j] = result.temp;
                            else data0[i * room_size + j] = result.temp;
                        }
                        else{
                            if(current_buffer == 0) data1[i*room_size + j] = data0[i*room_size + j];
                            else data0[i*room_size + j] = data1[i*room_size + j];
                        }
                    }
                }
                current_buffer = !current_buffer;
            }
    }

    (*CUDA_finish) = stabilized;
}

int main(int argc, char **argv) {
    // UNUSED(argc, argv);
    int threadnum = atoi(argv[argc-1]);
    printf("thread numeber is: %d \n", threadnum);

    bool first = true;
    bool finished = false;
    static hdist::State current_state, last_state;
    static std::chrono::high_resolution_clock::time_point begin, end;
    static const char* algo_list[2] = { "jacobi", "sor" };
    graphic::GraphicContext context{"Assignment 4"};
    auto grid = hdist::Grid{
            static_cast<size_t>(current_state.room_size),
            current_state.border_temp,
            current_state.source_temp,
            static_cast<size_t>(current_state.source_x),
            static_cast<size_t>(current_state.source_y)};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 4", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
        ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10, "%f");
        ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 100, "%f");
        ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 100, "%f");
        ImGui::DragInt("Source X", &current_state.source_x, 1, 1, current_state.room_size - 2, "%d");
        ImGui::DragInt("Source Y", &current_state.source_y, 1, 1, current_state.room_size - 2, "%d");
        ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1, "%f");
        ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo), algo_list, 2);

        if (current_state.algo == hdist::Algorithm::Sor) {
            ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0, 20.0, "%f");
        }

        if (current_state.room_size != last_state.room_size) {
            grid = hdist::Grid{
                    static_cast<size_t>(current_state.room_size),
                    current_state.border_temp,
                    current_state.source_temp,
                    static_cast<size_t>(current_state.source_x),
                    static_cast<size_t>(current_state.source_y)};
            first = true;
        }

        /* rewrite this part for mac user */
        if (current_state.room_size != last_state.room_size || current_state.block_size != last_state.block_size ||
            current_state.source_temp != last_state.source_temp || current_state.border_temp != last_state.border_temp ||
            current_state.tolerance != last_state.tolerance || current_state.sor_constant != last_state.sor_constant || 
            current_state.source_x != last_state.source_x || current_state.source_y != last_state.source_y) {
            
            last_state.room_size = current_state.room_size;
            last_state.block_size = current_state.block_size;
            last_state.source_temp = current_state.source_temp;
            last_state.border_temp = current_state.border_temp;
            last_state.tolerance = current_state.tolerance;
            last_state.source_x = current_state.source_x;
            last_state.source_y = current_state.source_y;
            finished = false;
        }


        if (first) {
            first = false;
            finished = false;
            begin = std::chrono::high_resolution_clock::now();
        }

        //////////// CODE HERE //////////////
        if (!finished) {

            int current_buffer = grid.current_buffer_num();
            double * device_data0;
            double * device_data1;
            int    * CUDA_finish;

            cudaError_t MallocStatus;

            /* malloc the grid and the state for cuda device to use */
            int size = current_state.room_size * current_state.room_size;
            cudaMallocManaged(&device_data0,sizeof(double) * size);
            cudaMallocManaged(&device_data1,sizeof(double) * size);
            cudaMallocManaged(&CUDA_finish, sizeof(int)    *  1  );

            MallocStatus =  cudaGetLastError();
            if (MallocStatus != cudaSuccess){
                fprintf(stderr, "malloc falied! : %s\n",
                        cudaGetErrorString(MallocStatus));
                return 0;
            }


            /* copy the data from host to device */
            for (int i = 0; i < grid.data0.size(); i++){
                device_data0[i] = grid.data0[i];
                device_data1[i] = grid.data1[i];
            }
            (*CUDA_finish) = 0;


            /* use to pass the int value to cuda kernel */
            int Algo; 
            if (current_state.algo == hdist::Algorithm::Jacobi ) Algo = 0;
            else Algo = 1;

            /* lauch the kernel */
            cudaError_t cudaStatus;
            mykernel<<<1, threadnum>>>(current_state.room_size, current_state.source_x, current_state.source_y, current_state.source_temp, current_state.border_temp,
                                        current_state.tolerance, current_state.sor_constant, Algo , current_buffer, threadnum, device_data0, device_data1, CUDA_finish );
            


            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "mykernel launch failed: %s\n",
                        cudaGetErrorString(cudaStatus));
                return 0;
            }

            /* synchronize the cuda thread */
            cudaDeviceSynchronize();

            /* copy the result back to the host */
            finished = (*CUDA_finish);
            for(int i = 0; i < grid.data0.size(); i++){
                grid.data0[i] = device_data0[i] ;
                grid.data1[i] = device_data1[i] ;
            }
            if (Algo == 0) grid.switch_buffer(); // Jacobi need to switch the buffer for each round 


            /* free the cuda memory */
            cudaFree(device_data0);
            cudaFree(device_data1);
            cudaFree(CUDA_finish);

            if (finished) end = std::chrono::high_resolution_clock::now();
        } else {
            ImGui::Text("stabilized in %lld ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        }
        //////////// CODE END //////////////

        const ImVec2 p = ImGui::GetCursorScreenPos();
        float x = p.x + current_state.block_size, y = p.y + current_state.block_size;
        for (size_t i = 0; i < current_state.room_size; ++i) {
            for (size_t j = 0; j < current_state.room_size; ++j) {
                auto temp = grid[{i, j}];
                auto color = temp_to_color(temp);
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state.block_size, y + current_state.block_size), color);
                y += current_state.block_size;
            }
            x += current_state.block_size;
            y = p.y + current_state.block_size;
        }
        ImGui::End();
    });
}
