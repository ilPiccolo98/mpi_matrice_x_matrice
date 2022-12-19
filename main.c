#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#define MATRIX_ROWS 6
#define MATRIX_COLUMNS 6
#define INDEX_GRID_ROWS 1
#define INDEX_GRID_COLUMNS 2
#define INDEX_SIZE_MATRICES 3
#define INDEX_HIDE_INPUT_MATRICES 4
#define INDEX_HIDE_OUTPUT_MATRIX 5
#define MIN_RANDOM_NUMBER 100
#define MAX_RANDOM_NUMBER -100

void print_matrix(float *matrix, int rows, int columns)
{
    int row, column;
    for(row = 0; row != rows; ++row)
    {
        printf("Row: %d\n", row + 1);
        for(column = 0; column != columns; ++column)
            printf("%f ", matrix[row * columns + column]);
        puts("");
    }
}

void create_grid(MPI_Comm *grid, MPI_Comm *grid_rows, MPI_Comm *grid_columns, int rank_processor, int rows, int columns, int *coordinates)
{
    int dim, *ndim, reorder, *period, vc[2];
    dim = 2;
    ndim = (int*) calloc (dim, sizeof(int));
    ndim[0] = rows;
    ndim[1] = columns;
    period = (int*) calloc (dim, sizeof(int));
    period[0] = period [1] = 0;
    reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, dim, ndim, period, reorder, grid);
    MPI_Cart_coords (*grid, rank_processor, 2, coordinates);
    vc[0] = 0;
    vc[1] = 1;
    MPI_Cart_sub(*grid, vc, grid_rows);
    vc[0] = 1;
    vc[1] = 0;
    MPI_Cart_sub(*grid, vc, grid_columns);
}

void free_matrix(int **matrix, int rows)
{
    int row;
    for(row = 0; row != rows; ++row)
        free(matrix[row]);
    free(matrix);
}

float get_random_number(float min, float max)
{
    float scale = rand() / (float) RAND_MAX;
	return min + scale * (max - (min));
    return rand() % 20;
}

float* init_matrix(int rows, int columns)
{
    float *matrix = (float*)calloc(rows * columns, sizeof(float));
    int row, column;
    for(row = 0; row != rows; ++row)
        for(column = 0; column != columns; ++column)
            matrix[row * columns + column] = get_random_number(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
    return matrix;
}

int get_grid_rows_from_argv(int argc, char *argv[])
{
    if(argc < INDEX_GRID_ROWS)
    {
        fprintf(stderr, "Error! Not enough argv arguments\n");
        exit(-1);
    }
    int grid_rows = atoi(argv[INDEX_GRID_ROWS]);
    if(grid_rows <= 0)
    {
        fprintf(stderr, "Error! Grid rows incorrect value\n");
        exit(-1);    
    }
    return grid_rows;
}

int get_grid_columns_from_argv(int argc, char *argv[])
{
    if(argc < INDEX_GRID_COLUMNS)
    {
        fprintf(stderr, "Error! Not enough argv arguments\n");
        exit(-1);
    }
    int grid_columns = atoi(argv[INDEX_GRID_COLUMNS]);
    if(grid_columns <= 0)
    {
        fprintf(stderr, "Error! Grid columns incorrect value\n");
        exit(-1);    
    }
    return grid_columns;
}

int get_size_matrix_from_argv(int argc, char *argv[])
{
    if(argc < INDEX_SIZE_MATRICES)
    {
        fprintf(stderr, "Error! Not enough argv arguments\n");
        exit(-1);
    }
    int size_matrices = atoi(argv[INDEX_SIZE_MATRICES]);
    if(size_matrices <= 1)
    {
        fprintf(stderr, "Error! Size matrices incorrect value\n");
        exit(-1); 
    }
    return size_matrices;
}

int get_hide_input_matrices_from_argv(int argc, char *argv[])
{
    if(argc < INDEX_HIDE_INPUT_MATRICES)
    {
        fprintf(stderr, "Error! Not enough argv arguments\n");
        exit(-1);
    }
    int hide_input_matrices = atoi(argv[INDEX_HIDE_INPUT_MATRICES]);
    if(hide_input_matrices != 0 && hide_input_matrices != 1)
    {
        fprintf(stderr, "Error! hide_input_matrices must have as value 0 or 1\n");
        exit(-1);
    }
    return hide_input_matrices;
}

int get_hide_output_matrix_from_argv(int argc, char *argv[])
{
    if(argc < INDEX_HIDE_OUTPUT_MATRIX)
    {
        fprintf(stderr, "Error! Not enough argv arguments\n");
        exit(-1);
    }
    int hide_output_matrix = atoi(argv[INDEX_HIDE_OUTPUT_MATRIX]);
    if(hide_output_matrix != 0 && hide_output_matrix != 1)
    {
        fprintf(stderr, "Error! hide_output_matrix must have as value 0 or 1\n");
        exit(-1);
    }
    return hide_output_matrix;
}

void check_correctness_grid_rows_columns(int processors, int grid_rows, int grid_columns)
{
    if(grid_rows * grid_columns != processors)
    {
        fprintf(stderr, "Error! grid_rows * grid_columns != processors\n");
        exit(-1);
    }
}

void send_submatrix(int rank_processor, int processors, int processor_destination, float *matrix_entry_point, int size_matrices, int size_submatrices, MPI_Comm comm)
{
    MPI_Datatype submatrix;
    MPI_Type_vector(size_submatrices, size_submatrices, size_matrices, MPI_FLOAT, &submatrix);
    MPI_Type_commit(&submatrix);
    MPI_Send(matrix_entry_point, 1, submatrix, processor_destination, 10 + processor_destination, comm);
}

int* get_matrix_entry_points(int processors, float *matrix, int size_matrices, int size_submatrices, MPI_Comm grid)
{
    int i;
    int *entry_points = (int*)calloc(processors, sizeof(int));
    entry_points[0] = 0;
    int coordinates[2];
    for(i = 1; i != processors; ++i)
    {
        MPI_Cart_coords(grid, i, 2, coordinates);
        int column = coordinates[1] * size_submatrices;
        int row = coordinates[0] * size_submatrices;
        entry_points[i] = row * size_matrices + column;
    }
    return entry_points;
}

void distribute_submatrices(int rank_processor, int processors, float *matrix, int size_matrices, float *submatrix, int size_submatrices, int grid_columns, MPI_Comm grid)
{
    if(rank_processor == 0)
    {
        int *entry_points = get_matrix_entry_points(processors, matrix, size_matrices, size_submatrices, grid);
        int i;
        for(i = 1; i != processors; ++i)
            send_submatrix(rank_processor, processors, i, matrix + entry_points[i], size_matrices, size_submatrices, grid);
        free(entry_points);
    }
    else
        MPI_Recv(submatrix, size_submatrices * size_submatrices, MPI_FLOAT, 0, 10 + rank_processor, grid, MPI_STATUS_IGNORE);
}

float* get_submatrix_processor_root(float *matrix, int size_matrix, int size_submatrix)
{
    float *submatrix = (float*)calloc(size_submatrix * size_submatrix, sizeof(float));
    int row, column;
    for(row = 0; row != size_submatrix; ++row)
        for(column = 0; column != size_submatrix; ++column)
            submatrix[row * size_submatrix + column] = matrix[row * size_matrix + column];
    return submatrix;
}

void execute_matrix_multiplication(float *matrix_a, float *matrix_b, float *destination, int size_matrix)
{
    int row, column, column_destination;
    for (row = 0; row < size_matrix; ++row)
        for (column = 0; column < size_matrix; ++column)
            for (column_destination = 0; column_destination < size_matrix; ++column_destination)
                destination[row * size_matrix + column_destination] += matrix_a[row * size_matrix + column] * matrix_b[column * size_matrix + column_destination];
}

float* get_block_partial_product(float *submatrix_a, float *submatrix_b, int size_submatrix, int grid_rows, int grid_columns, int *coordinates, MPI_Comm grid_rows_comm, MPI_Comm grid_columns_comm)
{
    float *submatrix_block = (float*)calloc(size_submatrix * size_submatrix, sizeof(float));
    int sender_coordinates[2];
    int sender_id = 0;
    sender_coordinates[0] = coordinates[0];
    int destination_submatrix_b[2];
    destination_submatrix_b[0] = (coordinates[0] + grid_columns - 1) % grid_columns;
    destination_submatrix_b[1] = coordinates[1];
    int destination_submatrix_b_id = 0;
    MPI_Cart_rank(grid_columns_comm, destination_submatrix_b, &destination_submatrix_b_id);
    int sender_submatrix_b[2];
    sender_submatrix_b[0] = (coordinates[0] + 1) % grid_columns;
    sender_submatrix_b[1] = coordinates[1];
    int sender_submatrix_b_id = 0;
    MPI_Cart_rank(grid_columns_comm, sender_submatrix_b, &sender_submatrix_b_id);
    float *submatrix_received = (float*)calloc(size_submatrix * size_submatrix, sizeof(float));
    int i;
    for(i = 0; i < grid_rows; i++)
    {
        sender_coordinates[1] = (coordinates[0] + i) % grid_rows;
        if(i == 0)
        {         
            if(coordinates[0] == coordinates[1])
            {
                sender_coordinates[1] = coordinates[1];
                memcpy(submatrix_received, submatrix_a, size_submatrix * size_submatrix * sizeof(float));
            }
            MPI_Cart_rank(grid_rows_comm, sender_coordinates, &sender_id);
            MPI_Bcast(submatrix_received, size_submatrix * size_submatrix, MPI_FLOAT, sender_id, grid_rows_comm);
            execute_matrix_multiplication(submatrix_received, submatrix_b, submatrix_block, size_submatrix);
        } 
        else 
        {          
            if(coordinates[1] == sender_coordinates[1])
                memcpy(submatrix_received, submatrix_a, size_submatrix * size_submatrix * sizeof(float));
            sender_id = (sender_id + 1) % grid_rows;
            MPI_Bcast(submatrix_received, size_submatrix * size_submatrix, MPI_FLOAT, sender_id, grid_rows_comm);
            MPI_Send(submatrix_b, size_submatrix * size_submatrix, MPI_FLOAT, destination_submatrix_b_id, 30, grid_columns_comm);
            MPI_Recv(submatrix_b, size_submatrix * size_submatrix, MPI_FLOAT, sender_submatrix_b_id, 30, grid_columns_comm, MPI_STATUS_IGNORE);
            execute_matrix_multiplication(submatrix_received, submatrix_b, submatrix_block, size_submatrix);
        }
    }
    free(submatrix_received);
    return submatrix_block;
}

void copy_block(float *matrix, int size_matrix, float *block, int size_block, int row_stride, int column_stride)
{
    int row_matrix, row_block, column;
    for(row_matrix = row_stride, row_block = 0; row_block < size_block; ++row_matrix, ++row_block)
        for(column = 0; column < size_block; ++column)
            matrix[row_matrix * size_matrix + column + column_stride] = block[row_block * size_block + column];
}

void get_multiplication_product(int rank_processor, int processors, float *product_matrix, float size_product_matrix, float *submatrix_block, float size_submatrix, MPI_Comm grid_comm)
{
    if(rank_processor == 0)
    {
        copy_block(product_matrix, size_product_matrix, submatrix_block, size_submatrix, 0, 0);
        float *received_block = (float*)calloc(size_submatrix * size_submatrix, sizeof(float));
        int coordinates[2];
        int i;
        for(i = 1; i != processors; ++i)
        {
            MPI_Cart_coords (grid_comm, i, 2, coordinates);
            MPI_Recv(received_block, size_submatrix * size_submatrix, MPI_FLOAT, i, 100 + i, grid_comm, MPI_STATUS_IGNORE);
            copy_block(product_matrix, size_product_matrix, received_block, size_submatrix, coordinates[0] * size_submatrix, coordinates[1] * size_submatrix);
        }
        free(received_block);
    }
    else
        MPI_Send(submatrix_block, size_submatrix * size_submatrix, MPI_FLOAT, 0, 100 + rank_processor, grid_comm);
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    int rank_processor = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_processor);
    int processors = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    int grid_rows = get_grid_rows_from_argv(argc, argv);
    int grid_columns = get_grid_columns_from_argv(argc, argv);
    int size_matrices = get_size_matrix_from_argv(argc, argv);
    int hide_input_matrices = get_hide_input_matrices_from_argv(argc, argv);
    int hide_output_matrix = get_hide_output_matrix_from_argv(argc, argv);
    check_correctness_grid_rows_columns(processors, grid_rows, grid_columns);
    MPI_Bcast(&grid_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size_matrices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    float *matrix_a = NULL;
    float *matrix_b = NULL;
    if(rank_processor == 0)
    {
        matrix_a = init_matrix(size_matrices, size_matrices);
        matrix_b = init_matrix(size_matrices, size_matrices);
    }
    int size_submatrices = size_matrices / grid_rows;
    float *submatrix_a = NULL;
    float *submatrix_b = NULL;
    if(rank_processor == 0)
    {
        submatrix_a = get_submatrix_processor_root(matrix_a, size_matrices, size_submatrices);
        submatrix_b = get_submatrix_processor_root(matrix_b, size_matrices, size_submatrices);
    }
    else
    {
        submatrix_a = (float*)calloc(size_submatrices * size_submatrices, sizeof(float));
        submatrix_b = (float*)calloc(size_submatrices * size_submatrices, sizeof(float));
    }
    if(!hide_input_matrices && rank_processor == 0)
    {
        puts("---------------------------------------------");
        puts("Matrix A:");
        print_matrix(matrix_a, size_matrices, size_matrices);
        puts("---------------------------------------------");
        puts("Matrix B:");
        print_matrix(matrix_b, size_matrices, size_matrices);
    }
    MPI_Comm grid_comm;
    MPI_Comm grid_rows_comm;
    MPI_Comm grid_columns_comm;
    int coordinates[2];
    create_grid(&grid_comm, &grid_rows_comm, &grid_columns_comm, rank_processor, grid_rows, grid_columns, coordinates);
    int grid_rank_processor = 0;
    MPI_Comm_rank(grid_comm, &grid_rank_processor);
    int grid_processors = 0;
    MPI_Comm_size(grid_comm, &grid_processors);
    distribute_submatrices(rank_processor, processors, matrix_a, size_matrices, submatrix_a, size_submatrices, grid_columns, grid_comm);
    distribute_submatrices(rank_processor, processors, matrix_b, size_matrices, submatrix_b, size_submatrices, grid_columns, grid_comm);
    double time_beginning = MPI_Wtime();
    float *submatrix_block = get_block_partial_product(submatrix_a, submatrix_b, size_submatrices, grid_rows, grid_columns, coordinates, grid_rows_comm, grid_columns_comm);
    double time_end = MPI_Wtime();
    double elapsed_time = time_end - time_beginning;
    double max_elapsed_time = 0;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    float *multiplication_product = NULL;
    if(rank_processor == 0)
        multiplication_product = (float*)calloc(size_matrices * size_matrices, sizeof(float));
    get_multiplication_product(grid_rank_processor, processors, multiplication_product, size_matrices, submatrix_block, size_submatrices, grid_comm);
    if(rank_processor == 0)
    {
        if(!hide_output_matrix)
        {
            puts("---------------------------------------------");
            puts("Result:");
            print_matrix(multiplication_product, size_matrices, size_matrices);
        }
        puts("---------------------------------------------");
        printf("Time elapsed: %f\n", max_elapsed_time);
        free(multiplication_product);
    }
    free(matrix_a);
    free(matrix_b);
    free(submatrix_block);
    if(rank_processor != 0)
    {
        free(submatrix_a);
        free(submatrix_b);
    }
    MPI_Finalize();
}
