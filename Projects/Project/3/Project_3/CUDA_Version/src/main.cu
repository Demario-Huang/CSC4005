#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <stdio.h>
#include <chrono>
#include <iostream>

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

/* set a mutex to avoid data race */
__device__ __managed__ int culock[100];

__device__ void CU_lock(size_t i ){
    bool flag = true;
    do{
        flag = atomicCAS(& ( culock[i] ),0,1);
        __threadfence_system();
    }while(flag);
}

__device__ void CU_unlock(size_t i){
    atomicCAS(&(culock[i]),1,0);
    __threadfence_system();
}

/* rewrite the function for cuda to compute */
__device__ void CU_check_and_update(size_t i, size_t j, double radius, double gravity, 
                                    double * device_m, double * device_x,double * device_y ,double * device_vx, double * device_vy, double * device_ax, double * device_ay ){

    auto delta_x = device_x[i] - device_x[j];
    auto delta_y = device_y[i] - device_y[j];
    auto distance_square = delta_x*delta_x + delta_y * delta_y;
    double COLLISION_RATIO = 0.01;
    auto ratio = 1 + COLLISION_RATIO;
    if (distance_square < radius * radius) {
        distance_square = radius * radius;
    }
    auto distance = sqrt(distance_square);
    if (distance < radius) {
        distance = radius;
    }
    if ( distance_square < (radius*radius) ) {
        auto dot_prod = delta_x * ( device_vx[i] - device_vx[j] )
                        + delta_y * (device_vy[i] - device_vy[j] );
        auto scalar = 2 / (device_m[i] + device_m[j] ) * dot_prod / distance_square;
        device_vx[i] -= scalar * delta_x * device_m[j];
        device_vy[i] -= scalar * delta_y * device_m[j];
        device_vx[j] += scalar * delta_x * device_m[i];
        device_vy[j] += scalar * delta_y * device_m[i];
        // now relax the distance a bit: after the collision, there must be
        // at least (ratio * radius) between them
        device_x[i] += delta_x / distance * ratio * radius / 2.0;
        device_y[i] += delta_y / distance * ratio * radius / 2.0;
        device_x[j] -= delta_x / distance * ratio * radius / 2.0;
        device_y[j] -= delta_y / distance * ratio * radius / 2.0;
    } else {
        // update acceleration only when no collision
        auto scalar = gravity / distance_square / distance;
        device_ax[i] -= scalar * delta_x * device_m[j];
        device_ay[i] -= scalar * delta_y * device_m[j];
        device_ax[j] += scalar * delta_x * device_m[i];
        device_ay[j] += scalar * delta_y * device_m[i];
    }
} 

/* rewrite the function for cuda to compute */
__device__ void CU_handle_wall_collision(size_t i, double position_range, double radius, 
                                         double * device_m, double * device_x,double * device_y ,double * device_vx, double * device_vy, double * device_ax, double * device_ay ){
    bool flag = false;
    double COLLISION_RATIO = 0.01;

    if (device_x[i] <= radius) {
        flag = true;
        device_x[i] = radius + radius * COLLISION_RATIO;
        device_vx[i] = -device_vx[i];
    } else if (device_x[i] >= position_range - radius) {
        flag = true;
        device_x[i] = position_range - radius - radius * COLLISION_RATIO;
        device_vx[i] = -device_vx[i];
    }

    if (device_y[i] <= radius) {
        flag = true;
        device_y[i] = radius + radius * COLLISION_RATIO;
        device_vy[i] = -device_vy[i];
    } else if (device_y[i] >= position_range - radius) {
        flag = true;
        device_y[i] = position_range - radius - radius * COLLISION_RATIO;
        device_vy[i] = -device_vy[i];
    }
    if (flag) {
        device_ax[i] = 0;
        device_ay[i] = 0;
    }

}

/* rewrite the function for cuda to compute */
__device__ void CU_update_for_tick(size_t i , double elapse, double space, double radius,
                                    double * device_m, double * device_x,double * device_y ,double * device_vx, double * device_vy, double * device_ax, double * device_ay ){

        device_vx[i] += device_ax[i] * elapse;
        device_vy[i] += device_ay[i] * elapse;
        CU_handle_wall_collision(i, space, radius,device_m, device_x, device_y, device_vx, device_vy ,device_ax , device_ay);
        device_x[i] += device_vx[i]  * elapse;
        device_y[i] += device_vy[i]  * elapse;
        CU_handle_wall_collision(i, space, radius,device_m, device_x, device_y, device_vx, device_vy ,device_ax , device_ay);
}


__global__ void mykernel(double space,double gravity,double radius, double elapse,double max_mass, int bodies, size_t size, int threadnum ,
                         double * device_m, double * device_x,double * device_y ,double * device_vx, double * device_vy, double * device_ax, double * device_ay ){
                        
    
    /* calculate the lenghth for each rank */                     
    int rank = threadIdx.x;

    int mylen = size / threadnum;
    int remain = size % (threadnum);
    if (rank < remain) mylen++;

    size_t min;
    size_t max;

    if (rank < remain) min = mylen * rank;
    else min = remain * (mylen+1) + (rank - remain) * mylen;
    max = min + mylen;
    
    /* parallel computation part, rewrite the function for cuda thread to compute  */
    for ( size_t i = min; i < max; ++i){
        for (size_t j = i + 1; j < size; j++){
            CU_lock(i);
            CU_lock(j);
            CU_check_and_update(i,j, radius, gravity, device_m, device_x,device_y,device_vx, device_vy, device_ax, device_ay);
            CU_unlock(i);
            CU_unlock(j);     

    }

    /* synchronization for cuda thread to avoid data race */
    __syncthreads(); 
    
    for ( size_t i = min;  i < max; ++i){
            CU_update_for_tick(i, elapse, space, radius, device_m, device_x, device_y, device_vx, device_vy, device_ax, device_ay);
    }
    

}



int main(int argc, char **argv) {
    // UNUSED(argc, argv);

    int threadnum = atoi(argv[argc-1]);
    printf("thread number is: %d \n", threadnum);

    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 20;
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

            //// CODE HERE //////
            {
                using namespace std::chrono;
                auto begin = std::chrono::high_resolution_clock::now();

                /* create array that store the value in the device */
                double * device_m; 
                double * device_x;
                double * device_y;
                double * device_vx;
                double * device_vy;
                double * device_ax;
                double * device_ay;

                cudaError_t MallocStatus;

                /* malloc the memory for device variable and accessible for both host and device */ 
                cudaMallocManaged(&device_m,sizeof(double) * pool.m.size());
                cudaMallocManaged(&device_x,sizeof(double)* pool.m.size());
                cudaMallocManaged(&device_y,sizeof(double)* pool.m.size());
                cudaMallocManaged(&device_vx,sizeof(double)* pool.m.size());
                cudaMallocManaged(&device_vy,sizeof(double)* pool.m.size());
                cudaMallocManaged(&device_ax,sizeof(double)* pool.m.size());
                cudaMallocManaged(&device_ay,sizeof(double)* pool.m.size());
                
                MallocStatus =  cudaGetLastError();
                if (MallocStatus != cudaSuccess){
                    printf("malloc memory failed! please check your device\n");
                    fprintf(stderr, "malloc falied! : %s\n",
                            cudaGetErrorString(MallocStatus));
                    return 0;
                }

                /* pass the value from host to the device to calculate */
                for ( size_t i = 0; i < pool.size(); i++){
                    device_m[i] = pool.m[i];
                    device_x[i] = pool.x[i];
                    device_y[i] = pool.y[i];
                    device_vx[i] = pool.vx[i];
                    device_vy[i] = pool.vy[i];
                    device_ax[i] = pool.ax[i];
                    device_ay[i] = pool.ay[i];
                }

                
                /* cuda thread for calculation */
                cudaError_t cudaStatus;
                mykernel<<<1, threadnum>>>(space,gravity,radius,elapse,max_mass, bodies, pool.size(), threadnum, 
                                            device_m, device_x, device_y,device_vx, device_vy, device_ax, device_ay );

                
                /* if error then terminated */
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "mykernel launch failed: %s\n",
                            cudaGetErrorString(cudaStatus));
                    return 0;
                }
                

                /* after calculation, abort the cuda thread */
                cudaDeviceSynchronize();
                

                /* when calculation done, pass the value from device to host */
                for (size_t i = 0; i < pool.size(); i++ ){
                    pool.m[i] = device_m[i];
                    pool.x[i] =  device_x[i];
                    pool.y[i] = device_y[i];
                    pool.vx[i] = device_vx[i];
                    pool.vy[i] = device_vy[i];
                    pool.ax[i] = device_ax[i];
                    pool.ay[i] = device_ay[i];
                }
                
                /* free the variable for next use */
                cudaFree(device_m);
                cudaFree(device_x);
                cudaFree(device_y);
                cudaFree(device_vx);
                cudaFree(device_vy);
                cudaFree(device_ax);
                cudaFree(device_ay);


                auto end = std::chrono::high_resolution_clock::now();

                countbodies += bodies;
                duration += duration_cast<nanoseconds>(end - begin).count();

                /* print the speed: ms per body */
                if (countbodies >= THRESHOLD){
                    std::cout << "The speed is (time per body): " << duration/countbodies << " ms" << std::endl;
                    countbodies = 0;
                    duration = 0;
                }
            }
            //// CODE END //////
            

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


