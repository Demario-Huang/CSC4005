//
// Created by schrodinger on 11/2/21.
//
#pragma once
#include <iostream>
#include <random>
#include <utility>
#include <pthread.h>
//#include "/usr/local/opt/libomp/include/omp.h"


class BodyPool;

struct threadpara
{
    int thread_id;
    int thread_num;
    double elapse;
    double gravity;
    double position_range;
    double radius;
    BodyPool * pool;
};


#define MAX_THREADS 128
threadpara thread_paras[MAX_THREADS];
pthread_attr_t attr;
pthread_mutex_t mutex[100];

class BodyPool {
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> ax;
    std::vector<double> ay;
    std::vector<double> m;
    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.
    static constexpr double COLLISION_RATIO = 0.01;
public:

    class Body {
        size_t index;
        BodyPool &pool;

        friend class BodyPool;

        Body(size_t index, BodyPool &pool) : index(index), pool(pool) {}

    public:
        double &get_x() {
            return pool.x[index];
        }

        double &get_y() {
            return pool.y[index];
        }

        double &get_vx() {
            return pool.vx[index];
        }

        double &get_vy() {
            return pool.vy[index];
        }

        double &get_ax() {
            return pool.ax[index];
        }

        double &get_ay() {
            return pool.ay[index];
        }

        double &get_m() {
            return pool.m[index];
        }

        double distance_square(Body &that) {
            auto delta_x = get_x() - that.get_x();
            auto delta_y = get_y() - that.get_y();
            return delta_x * delta_x + delta_y * delta_y;
        }

        double distance(Body &that) {
            return std::sqrt(distance_square(that));
        }

        double delta_x(Body &that) {
            return get_x() - that.get_x();
        }

        double delta_y(Body &that) {
            return get_y() - that.get_y();
        }

        bool collide(Body &that, double radius) {
            return distance_square(that) <= radius * radius;
        }

        // collision with wall
        void handle_wall_collision(double position_range, double radius) {
            bool flag = false;
            if (get_x() <= radius) {
                flag = true;
                get_x() = radius + radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            } else if (get_x() >= position_range - radius) {
                flag = true;
                get_x() = position_range - radius - radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            }

            if (get_y() <= radius) {
                flag = true;
                get_y() = radius + radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            } else if (get_y() >= position_range - radius) {
                flag = true;
                get_y() = position_range - radius - radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            }
            if (flag) {
                get_ax() = 0;
                get_ay() = 0;
            }
        }

        void update_for_tick(
                double elapse,
                double position_range,
                double radius) {
            get_vx() += get_ax() * elapse;
            get_vy() += get_ay() * elapse;
            handle_wall_collision(position_range, radius);
            get_x() += get_vx() * elapse;
            get_y() += get_vy() * elapse;
            handle_wall_collision(position_range, radius);
        }

    };

    BodyPool(size_t size, double position_range, double mass_range) :
            x(size), y(size), vx(size), vy(size), ax(size), ay(size), m(size) {
        std::random_device device;
        std::default_random_engine engine{device()};
        std::uniform_real_distribution<double> position_dist{0, position_range};
        std::uniform_real_distribution<double> mass_dist{0, mass_range};
        for (auto &i : x) {
            i = position_dist(engine);
        }
        for (auto &i : y) {
            i = position_dist(engine);
        }
        for (auto &i : m) {
            i = mass_dist(engine);
        }
    }

    Body get_body(size_t index) {
        return {index, *this};
    }

    void clear_acceleration() {
        ax.assign(m.size(), 0.0);
        ay.assign(m.size(), 0.0);
    }

    size_t size() {
        return m.size();
    }

    static void check_and_update(Body i, Body j, double radius, double gravity) {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius) {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius) {
            distance = radius;
        }
        if (i.collide(j, radius)) {
            auto dot_prod = delta_x * (i.get_vx() - j.get_vx())
                            + delta_y * (i.get_vy() - j.get_vy());
            auto scalar = 2 / (i.get_m() + j.get_m()) * dot_prod / distance_square;
            i.get_vx() -= scalar * delta_x * j.get_m();
            i.get_vy() -= scalar * delta_y * j.get_m();
            j.get_vx() += scalar * delta_x * i.get_m();
            j.get_vy() += scalar * delta_y * i.get_m();
            // now relax the distance a bit: after the collision, there must be
            // at least (ratio * radius) between them
            i.get_x() += delta_x / distance * ratio * radius / 2.0;
            i.get_y() += delta_y / distance * ratio * radius / 2.0;
            j.get_x() -= delta_x / distance * ratio * radius / 2.0;
            j.get_y() -= delta_y / distance * ratio * radius / 2.0;
        } else {
            // update acceleration only when no collision
            auto scalar = gravity / distance_square / distance;
            i.get_ax() -= scalar * delta_x * j.get_m();
            i.get_ay() -= scalar * delta_y * j.get_m();
            j.get_ax() += scalar * delta_x * i.get_m();
            j.get_ay() += scalar * delta_y * i.get_m();
        }
    }


    void update_for_tick(double elapse,
                         double gravity,
                         double position_range,
                         double radius,
                         int    threadnum) {
        ax.assign(size(), 0);
        ay.assign(size(), 0);

        /* create threads space */
        pthread_t *threads=(pthread_t*)malloc(sizeof(pthread_t)*threadnum);

        /* pass the value to each thread */
        for (int i=0; i<threadnum; i++){
            thread_paras[i].thread_id = i;
            thread_paras[i].thread_num = threadnum;
            thread_paras[i].elapse =elapse;
            thread_paras[i].gravity = gravity;
            thread_paras[i].position_range = position_range;
            thread_paras[i].radius = radius;
            thread_paras[i].pool = this;
        }

        /* initialize the pthread */
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

        /* initialize the mutex */
        for (int j = 0; j < 100; j++) {
            pthread_mutex_init(&mutex[j], NULL);
        }

        /* create the threads for calculation */
        for (int j = 0; j < threadnum; j++) {
            pthread_create(&threads[j],&attr,Pthread_update,(void*)&thread_paras[j]);
        }

        /* join the pthread synchronization */
        for (int i = 0; i < threadnum; i++){
            pthread_join(threads[i],NULL);
        }

        pthread_attr_destroy(&attr);
        free(threads);
        
    }

    /* function for parallel calcualtion */
    static void*  Pthread_update(void * t){
        
        /* pass the variable to local function */
        struct threadpara *recv_para = (struct threadpara*)t;
        int myid = (*recv_para).thread_id;
        int threadnum = (*recv_para).thread_num;
        double elapse = (*recv_para).elapse;
        double position_range = (*recv_para).position_range;
        double gravity = (*recv_para).gravity;
        double radius = (*recv_para).radius;
        BodyPool * pool = (*recv_para).pool;

        int mylen = pool->size() / threadnum;
        int remain = pool->size() % (threadnum);
        if (myid < remain) mylen++;

        size_t min;
        size_t max;

        if (myid < remain) min = mylen * myid;
        else min = remain * (mylen+1) + (myid - remain) * mylen;
        max = min + mylen;
        
        /* parallel computation part */ 
        for (size_t i = min; i < max; ++i) {
            for (size_t j = i + 1; j < pool->size(); ++j) {

                /* set mutex to avoid the data race */
                pthread_mutex_lock(&mutex[i]);
                pthread_mutex_lock(&mutex[j]);
                check_and_update(pool->get_body(i), pool->get_body(j), radius, gravity);
                pthread_mutex_unlock(&mutex[i]);
                pthread_mutex_unlock(&mutex[j]);
                
            }
        }

        for (size_t i = min; i < max; ++i) {
            pool->get_body(i).update_for_tick(elapse, position_range, radius);
        }
        pthread_exit(NULL);
    }
};


