#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <mpi.h>
#include <vector>


ImColor temp_to_color(double temp) {
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

int main(int argc, char **argv) {
    
    int rank;
    int num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    MPI_Datatype MPIparam;
    MPI_Type_contiguous(5, MPI_FLOAT,&MPIparam);
    MPI_Type_commit(&MPIparam);

    bool all_finished ;

    /* ROOT */
    if (0 == rank) {
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

            ///////////// CODE HERE //////////////
            if (!finished) {
                
                bool root_finish = false;

                /* used for bcast the value */
        
                int size = current_state.room_size * current_state.room_size;

                /* board cast the information to calcualte */
                MPI_Bcast(&current_state.border_temp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&current_state.source_temp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&current_state.block_size, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&current_state.tolerance, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&current_state.sor_constant, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

                MPI_Bcast(&current_state.room_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&current_state.source_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&current_state.source_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&current_state.algo, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&grid.get_current_buffer()[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                /* calculate the value */
                root_finish = hdist::calculate(current_state, grid, rank, num_processes);

                /* used to collect all the bool finish in every rank */
                MPI_Allreduce(&root_finish, &all_finished, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD );
                
                finished = all_finished;

                if (finished) end = std::chrono::high_resolution_clock::now();

            } else {
                ImGui::Text("stabilized in %lld ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
            }
            ///////////////// CODE END ///////////////

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

    else{

        /* Create a new state*/
        static hdist::State current_state;
        auto grid = hdist::Grid{
            static_cast<size_t>(current_state.room_size),
            current_state.border_temp,
            current_state.source_temp,
            static_cast<size_t>(current_state.source_x),
            static_cast<size_t>(current_state.source_y)};

        int room_size = 300;
        int source_x = 150;
        int source_y = 150;
        int algo = 0;
        float  block_size = 2; float source_temp = 100; float border_temp = 36; 
        float tolerance = 0.02; float sor_constant = 4.0;

        bool other_finished = false;

        while(true){
            
            /* Bacst the State information */
            MPI_Bcast(&border_temp, 1 , MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&source_temp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&block_size, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&tolerance, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&sor_constant, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

            MPI_Bcast(&room_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&source_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&source_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&algo, 1, MPI_INT, 0, MPI_COMM_WORLD);

            /* Update the state and grid */
            current_state.source_x = source_x; 
            current_state.source_y = source_y;
            current_state.room_size = room_size;
            current_state.border_temp = border_temp;
            current_state.source_temp = source_temp;
            current_state.block_size = block_size;
            current_state.tolerance = tolerance;
            current_state.sor_constant = sor_constant;
            if (algo == 0) current_state.algo = hdist::Algorithm::Jacobi;
            else current_state.algo =  hdist::Algorithm::Sor;

            grid = hdist::Grid{
                        static_cast<size_t>(current_state.room_size),
                        current_state.border_temp,
                        current_state.source_temp,
                        static_cast<size_t>(current_state.source_x),
                        static_cast<size_t>(current_state.source_y)};

            /* boardcast the temperature information */
            int size = current_state.room_size * current_state.room_size;
            MPI_Bcast(&grid.get_current_buffer()[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            other_finished = hdist::calculate(current_state, grid,rank,num_processes);
          
            MPI_Allreduce(&other_finished, &all_finished, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD );
        }
    }
    MPI_Finalize();

}
