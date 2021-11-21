#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <iterator>
#include <vector>
#include <set>
#include "mpi/mpi.h"
#include <string.h>


//#define DEBUG_MODE

float x_func (int x, int y) {
    return 10.0 * x + 1.5 * y;
}

float y_func (int x, int y) {
    return 1.5 * x + 10.0 * y;
}

/* Constructor in order to implement emplace-back */
//class Point {
//    float coord[2];
//    int index;
//public:
//    Point(int x, int y, int row_len): coord{x_func(x,y), y_func(x,y)}, index(y*row_len + x) {}
//};

typedef struct{
    float x_coord;
    float y_coord;
    int index;
} point;

const int nitems = 3;
int blocklengths[3] = {1, 1, 1};
MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_INT};
MPI_Datatype mpi_point_type;
MPI_Aint offsets[3];

/* Assume that every process takes a full row */
void init(point* array, int size, int n2, int rank) {
    for (int i = 0; i < size; i++) {
        int index = size * rank + i;
        int x_coord = index / n2;
        int y_coord = index % n2;
        array[i].x_coord = x_func(x_coord, y_coord);
        array[i].y_coord = y_func(x_coord, y_coord);
        array[i].index = index;
    }
}

bool compare(point& a, point& b) {
    if (a.y_coord > b.y_coord) {
        return true;
    } else {
        return false;
    }
}

void merge_left(point* temp_buffer, int personal_size, point* recv_buffer, point* array, int rank) {
    int ia = 0, ib = 0;

    for (int i = 0; i < personal_size; i++) {
//        if (rank == 0) {
//            std::cout << array[ia].y_coord, recv_buffer[ib]
//        }
        if (compare(array[ia], recv_buffer[ib])) {
            temp_buffer[i] = recv_buffer[ib];
            ib++;
        } else {
            temp_buffer[i] = array[ia];
            ia++;
        }
    }
    memcpy(array, temp_buffer, personal_size * sizeof(point));
}

void merge_right(point* temp_buffer, int personal_size, point* recv_buffer, point* array) {
    int ia = personal_size - 1, ib = personal_size - 1;

    for (int i = personal_size - 1; i >= 0; i--) {
        if (compare(array[ia], recv_buffer[ib])) {
            temp_buffer[i] = array[ia];
            ia--;
        } else {
            temp_buffer[i] = recv_buffer[ib];
            ib--;
        }
    }
    memcpy(array, temp_buffer, personal_size * sizeof(point));
}

/* сравнение двух целых */
int comp (point* a, point *b)
{
    if (a->y_coord > b->y_coord){
        return +1;
    }if (a->y_coord < b->y_coord){
        return -1;
    }
    return 0;
}

void BatcherSort(point *array, int length, int rank, int size, point *recv_buffer, point * temp_buffer, int personal_size) {
    std::set<int> current_tact;
    int t = (int) ceil(log2(size)) - 1;
    int p_initial = (int) pow(2, t);
    int p = p_initial;

    do {
        int q = p_initial;
        int r = 0;
        int d = p;

        do {
            if (r != 0) {
                d = q - p;
                q >>= 1;
            }

            for (int i = 0; i < size; ++i) {
                if (i < size - d and (i & p) == r) {
                    if (rank == i) {
                        if (rank == 0) {
                            std::cout<< "HERE" << std::endl;
                        }
                        //std::cout << "process rank №"<<rank << " will change with " << i + d << " on iter " << iter << std::endl;
                        //std::cout.flush();
                        MPI_Sendrecv(array, personal_size, mpi_point_type, i+d, 0, recv_buffer, personal_size,
                                     mpi_point_type, i+d, MPI_ANY_TAG, MPI_COMM_WORLD, nullptr);
                        merge_left(temp_buffer, personal_size, recv_buffer, array, rank);
                    }

                } else if (((i - d) & p) == r && i-d >= 0) {
                    if (rank == i) {
                        //std::cout << "process rank №"<< rank << " will change with " << i - d << " on iter " << iter <<std::endl;
                        std::cout.flush();
                        MPI_Sendrecv(array, personal_size, mpi_point_type, i-d, 0, recv_buffer, personal_size,
                                     mpi_point_type, i-d, MPI_ANY_TAG, MPI_COMM_WORLD, nullptr);
                        merge_right(temp_buffer, personal_size, recv_buffer, array);
                    }
                }
            }
            r = p;
        } while (q != p);
        p /= 2;
        MPI_Barrier(MPI_COMM_WORLD);
    } while (p > 0);
}

void print_array(point* array, int personal_size) {
    for (int i = 0; i < personal_size; i++) {
        std::cout << array[i].y_coord << " ";
    }
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char *str;
    int n1 = (int) strtol(argv[1], &str, 10);
    int n2 = (int) strtol(argv[2], &str, 10);

    offsets[0] = offsetof(point, x_coord);
    offsets[1] = offsetof(point, y_coord);
    offsets[2] = offsetof(point, index);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_point_type);
    MPI_Type_commit(&mpi_point_type);

    int personal_size = n1 * n2 / size;
    auto point_array = (point*) malloc(personal_size * sizeof(point));
    auto recv_buffer = (point*) malloc(personal_size * sizeof(point));
    auto temp_buffer = (point*) malloc(personal_size * sizeof(point));

    init(point_array, personal_size, n2, rank);
    qsort(point_array,personal_size, sizeof(point), (int(*) (const void *, const void *)) comp);


    BatcherSort(point_array, personal_size, rank, size, recv_buffer, temp_buffer, personal_size);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        print_array(point_array, personal_size);
    }

    free(point_array);
    free(recv_buffer);
    free(temp_buffer);

    MPI_Finalize();
}