#include <gsl/gsl_matrix.h>
extern double *row_sums, *col_sums, all_sum;
extern double lambda, mu, tau;  // thresholds for distinguishing roundoff between fault
extern double u;

int ft_dgemm(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C, int rank);
double infnorm(gsl_matrix *A);
double onenorm(gsl_matrix *A);
int verify_checksum(gsl_matrix *C);

void build_checksum(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C);
