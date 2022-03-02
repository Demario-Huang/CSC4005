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

/* used for store values and pass the value to each thread */
struct threadpara
{
   int thread_id;
   int thread_num;
};

/* declare the global variable */
#define MAX_THREADS 128
static int center_x = 0;
static int center_y = 0;
static int size = 800;
static int scale = 1;
static int k_value = 100;

/* used for dynamic allocation of Pthread */
static int count = 0;

/* declare the Pthread variable */
threadpara thread_paras[MAX_THREADS];
pthread_mutex_t cal_mutex;
pthread_attr_t attr;

/* used for draw the picture */
Square canvas(size);
static constexpr float MARGIN = 4.0f;
static constexpr float BASE_SPACING = 2000.0f;
static constexpr size_t SHOW_THRESHOLD = 500000000ULL;


void* Pthreadcal(void *t){

    /* pass the value to each thread */
    struct threadpara *recv_para = (struct threadpara*)t;
    int myid = (*recv_para).thread_id;
    int threadnum = (*recv_para).thread_num;
    
    
    double cx = static_cast<double>(size) / 2 + center_x;
    double cy = static_cast<double>(size) / 2 + center_y;
    double zoom_factor = static_cast<double>(size) / 4 * scale;

    /* set a original current start position    */
    /* when the thread finish this task, it will find next task to calculate */
    count = threadnum - 1;
    int current  = myid;

    while (count < size && current < size){

        for (int j = 0; j < size; ++j){
            double x = (static_cast<double>(j) - cx) / zoom_factor;
            double y = (static_cast<double>(current) - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < k_value);
            canvas[{current,j }] = k;
        }

        /* lock the count and update the task */
        pthread_mutex_lock(&cal_mutex);
        current = count + 1;
        count++;
        pthread_mutex_unlock(&cal_mutex);
        if (count >= size) break;
    }

    pthread_exit(NULL);
}



int main(int argc, char **argv) {
    int rank;

    int threadnum = atoi(argv[argc-1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (0 == rank) {
        graphic::GraphicContext context{"Assignment 2"};
        size_t duration = 0;
        size_t pixels = 0;
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

                    ///////// CODE HERE ///////////////
                    {
                        /* malloc the space for threads */
                        pthread_t *threads=(pthread_t*)malloc(sizeof(pthread_t)*threadnum);

                        /* pass the value to each thread */
                        for (int i=0; i<threadnum; i++){
                            thread_paras[i].thread_id = i;
                            thread_paras[i].thread_num = threadnum;
                        }

                        pthread_mutex_init(&cal_mutex,NULL);
                        pthread_attr_init(&attr);
                        pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

                        /* initialization */
                        for (int j = 0; j < threadnum; j++) {
                            pthread_create(&threads[j],&attr,Pthreadcal,(void*)&thread_paras[j]);
                        }

                        /* join the thread */
                        for (int i = 0; i < threadnum; i++){
                            pthread_join(threads[i],NULL);
                        }

                        pthread_attr_destroy(&attr);
                        free(threads);
                    }
                    ///////// CODE END ///////////////

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
