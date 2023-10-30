typedef struct {
    int rows;
    int columns;
    double** values;
 } Matrix;

Matrix* create_matrix(int rows, int columns);
void fill_matrix(Matrix *matrix, double value[]);  //-> for i -> ligne for j -> colonne
void print_matrix(Matrix *matrix);
