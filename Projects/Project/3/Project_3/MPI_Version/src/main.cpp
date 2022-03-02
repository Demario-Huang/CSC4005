#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <iostream>
#include <mpi.h>

/* used to store the value for boardcast */
struct params_float{
    float gravity;
    float space;
    float radius;
    float elapse;
    float max_mass;
}__attribute__((packed));

int main(int argc, char **argv) {

    int rank;
    int num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    /* used for calculate the speed */
    size_t duration = 0;
    int countbodies = 0;
    int THRESHOLD = 500;

    /* used for allreduce function to store the sum of every rank's value */
    std::vector<double> recv_m;
    std::vector<double> recv_x;
    std::vector<double> recv_y;
    std::vector<double> recv_vx;
    std::vector<double> recv_vy;
    std::vector<double> recv_ax;
    std::vector<double> recv_ay;

    /* used to copy the value from root in order to calculate the final variable  */
    std::vector<double> copy_m;
    std::vector<double> copy_x;
    std::vector<double> copy_y;
    std::vector<double> copy_vx;
    std::vector<double> copy_vy;
    std::vector<double> copy_ax;
    std::vector<double> copy_ay;

    /* create a new datatype for boardcast */
    MPI_Datatype MPIparam;
    MPI_Type_contiguous(5, MPI_FLOAT,&MPIparam);
    MPI_Type_commit(&MPIparam);
    
    /* ROOT */
    if ( 0 == rank ) {
        static float gravity = 100;
        static float space = 800;
        static float radius = 5;
        static int bodies = 50; 
        static float elapse = 0.001;
        static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
        static float max_mass = 50;
        static float current_space = space;
        static float current_max_mass = max_mass;
        static int current_bodies = bodies;
        BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
        graphic::GraphicContext context{"Assignment 2"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
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
            ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
            ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
            ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
            ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
            ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
            ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
            ImGui::ColorEdit4("Color", &color.x);
            if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
                space = current_space;
                bodies = current_bodies;
                max_mass = current_max_mass;
                pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
            }
            {
                const ImVec2 p = ImGui::GetCursorScreenPos();

                /////// CODE BEGIN /////////

                {
                    using namespace std::chrono;
                    auto begin = high_resolution_clock::now();
                    
                    /* pass the information to other rank */
                    params_float paras = {gravity,space,radius,elapse,max_mass};
                    
                    MPI_Bcast(&paras,1,MPIparam,0, MPI_COMM_WORLD);
                    MPI_Bcast(&bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);

                    /* boardcast to make every pool have the same value */
                    MPI_Bcast(&pool.x[0], pool.x.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.y[0], pool.y.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.vx[0], pool.vx.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.vy[0], pool.vy.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.ax[0], pool.ax.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.ay[0], pool.ay.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.m[0], pool.m.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);

                    /* resize the length for every vector */
                    copy_m.resize(pool.m.size());
                    copy_x.resize(pool.x.size());
                    copy_y.resize(pool.y.size());
                    copy_vx.resize(pool.vx.size());
                    copy_vy.resize(pool.vy.size());
                    copy_ax.resize(pool.ax.size());
                    copy_ay.resize(pool.ay.size());
                    
                    /* copy the value from root*/
                    for ( size_t i = 0; i < pool.size(); i++){
                        copy_m[i] = pool.m[i];
                        copy_x[i] = pool.x[i];
                        copy_y[i] = pool.y[i];
                        copy_vx[i] = pool.vx[i];
                        copy_vy[i] = pool.vy[i];
                        copy_ax[i] = pool.ax[i];      
                        copy_ay[i] = pool.ay[i];
                    }
 
                    /* calculate the acceleration, speed for root */
                    pool.update_for_tick( gravity, radius, rank, num_processes);
                    

                    /* resize the recv vector for allreduce function */
                    recv_m.resize(pool.m.size());
                    recv_x.resize(pool.x.size());
                    recv_y.resize(pool.y.size());
                    recv_vx.resize(pool.vx.size());
                    recv_vy.resize(pool.vy.size());
                    recv_ax.resize(pool.ax.size());
                    recv_ay.resize(pool.ay.size());

                    
                    /* sum all of the value from every rank and store in the vector */
                    MPI_Allreduce(&pool.x[0], &recv_x[0],pool.x.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.y[0], &recv_y[0],pool.y.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.vx[0], &recv_vx[0],pool.vx.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.vy[0], &recv_vy[0],pool.vy.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.ax[0], &recv_ax[0],pool.ax.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.ay[0], &recv_ay[0],pool.ay.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


                    /*  calculate and update the final value, since allreduce calculate the sum, hence we minus (n-1)*original value  */
                    for (size_t i = 0; i < pool.size(); i++){
                        pool.x[i] = recv_x[i] - (num_processes - 1) * copy_x[i];
                        pool.y[i] = recv_y[i] - (num_processes - 1) * copy_y[i];
                        pool.vx[i] = recv_vx[i] - (num_processes - 1) * copy_vx[i];
                        pool.vy[i] = recv_vy[i] - (num_processes - 1) * copy_vy[i];
                        pool.ax[i] = recv_ax[i];   
                        pool.ay[i] = recv_ay[i];
                    }
                    
                    /* copy the value for next calculate distance */
                    for ( size_t i = 0; i < pool.size(); i++){
                        copy_m[i] = pool.m[i];
                        copy_x[i] = pool.x[i];
                        copy_y[i] = pool.y[i];
                        copy_vx[i] = pool.vx[i];
                        copy_vy[i] = pool.vy[i];
                        copy_ax[i] = pool.ax[i];
                        copy_ay[i] = pool.ay[i];
                    }

                    /* boardcast all the value to calculate the distance */
                    MPI_Bcast(&pool.x[0], pool.x.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.y[0], pool.y.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.vx[0], pool.vx.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.vy[0], pool.vy.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.ax[0], pool.ax.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.ay[0], pool.ay.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pool.m[0], pool.m.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    
                    /* calculat the distance for root */
                    pool.update_distance(elapse,space, radius, rank, num_processes);

                    /* sum all of the value from every rank and store in the vector */
                    MPI_Allreduce(&pool.x[0], &recv_x[0],pool.x.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.y[0], &recv_y[0],pool.y.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.vx[0], &recv_vx[0],pool.vx.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.vy[0], &recv_vy[0],pool.vy.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.ax[0], &recv_ax[0],pool.ax.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&pool.ay[0], &recv_ay[0],pool.ay.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                   
                    /* assign the final value for root */
                    for (size_t i = 0; i < pool.size(); i++){
                        pool.x[i] = recv_x[i] - (num_processes - 1) * copy_x[i];
                        pool.y[i] = recv_y[i] - (num_processes - 1) * copy_y[i];
                        pool.vx[i] = recv_vx[i] - (num_processes - 1) * copy_vx[i];
                        pool.vy[i] = recv_vy[i] - (num_processes - 1) * copy_vy[i];
                    }

                    auto end = high_resolution_clock::now();

                    countbodies += bodies;
                    duration += duration_cast<nanoseconds>(end - begin).count();
                    
                    /* print the speed: ms per body */
                    if (countbodies >= THRESHOLD){
                        std::cout << "The speed is (time per body): " << duration/countbodies << " ms" << std::endl;
                        countbodies = 0;
                        duration = 0;
                    }
                
                }

                ////// CODE END ///////////


                for (size_t i = 0; i < pool.size(); ++i) {
                    auto body = pool.get_body(i);
                    auto x = p.x + static_cast<float>(body.get_x());
                    auto y = p.y + static_cast<float>(body.get_y());
                    draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                }
            }
            ImGui::End();
        });
    }

    else{

        /* initial value */
        float gravity = 100; float space = 800; float radius = 5; 
        float elapse = 0.001; float max_mass = 50; int bodies = 20;  

        float current_space = space;
        float current_max_mass = max_mass;
        int current_bodies = bodies;

        /* iniialize the bodypool for every rank */
        params_float paras = {gravity,space,radius,elapse,max_mass};
        BodyPool pool(static_cast<size_t>(bodies), space, max_mass);

        /* set a while loop for sychronization */
        while (true) {

            /* boardcast the value for synchronization */
            MPI_Bcast(&paras,1,MPIparam,0, MPI_COMM_WORLD);
            MPI_Bcast(&bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            /* update the bodypool */
            gravity = paras.gravity;  max_mass = paras.max_mass;  radius = paras.radius;  elapse = paras.elapse;  space = paras.space;
            if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
                current_space = space;
                current_bodies = bodies;
                current_max_mass = max_mass;
                pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
            }
            
            /* boardcast the value for every rank to make sure they calculate the same number */
            MPI_Bcast(&pool.x[0], pool.x.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.y[0], pool.y.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.vx[0], pool.vx.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.vy[0], pool.vy.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.ax[0], pool.ax.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.ay[0], pool.ay.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.m[0], pool.m.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);

            /* calculate the values for each rank */
            pool.update_for_tick( gravity,radius, rank, num_processes);

            /* resize the recv value */
            recv_m.resize(pool.m.size());
            recv_x.resize(pool.x.size());
            recv_y.resize(pool.y.size());
            recv_vx.resize(pool.vx.size());
            recv_vy.resize(pool.vy.size());
            recv_ax.resize(pool.ax.size());
            recv_ay.resize(pool.ay.size());

            /* allreduce function to calculate the sum of every value */
            MPI_Allreduce(&pool.x[0], &recv_x[0],pool.x.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.y[0], &recv_y[0],pool.y.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.vx[0], &recv_vx[0],pool.vx.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.vy[0], &recv_vy[0],pool.vy.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.ax[0], &recv_ax[0],pool.ax.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.ay[0], &recv_ay[0],pool.ay.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            /* boarcast again for next distance calcualation */
            MPI_Bcast(&pool.x[0], pool.x.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.y[0], pool.y.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.vx[0], pool.vx.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.vy[0], pool.vy.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.ax[0], pool.ax.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.ay[0], pool.ay.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pool.m[0], pool.m.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            /* calcualte the distance */
            pool.update_distance(elapse,space, radius, rank, num_processes);

            /* sum the value for every rank */
            MPI_Allreduce(&pool.x[0], &recv_x[0],pool.x.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.y[0], &recv_y[0],pool.y.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.vx[0], &recv_vx[0],pool.vx.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.vy[0], &recv_vy[0],pool.vy.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.ax[0], &recv_ax[0],pool.ax.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&pool.ay[0], &recv_ay[0],pool.ay.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        }

    }
    MPI_Finalize();
}
