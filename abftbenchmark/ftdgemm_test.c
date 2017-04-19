#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#ifndef GSL_RANGE_CHECK_OFF
#define GSL_RANGE_CHECK_OFF
#endif

#include "ftblas.h"

long int N, R, ER;
gsl_matrix *target;


void randomize_matrix(gsl_matrix *A)
{
    int i, j;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, clock());

    for (i = 0; i < A->size1; i++) 
        for (j = 0; j < A->size2; j++) 
            gsl_matrix_set(A, i, j, gsl_ran_gaussian_ziggurat(r, 1));

    gsl_rng_free(r);
}



void test_ft_dgemm()
{

    float t0, t1;
    int m, n, k, err;
    struct timeval tvs,tve;
    double span;
    gsl_matrix *A, *B, *C, *D;
    A = gsl_matrix_alloc(N, N);
    B = gsl_matrix_alloc(N, N);
    C = gsl_matrix_alloc(N, N);
    D = gsl_matrix_alloc(N, N);
    target = C;
    m = A->size1; n = B->size2; k = A->size2;
    randomize_matrix(A);
    randomize_matrix(B);
    // allocate workspace for verify_checksum
    row_sums = (double*)calloc(m, sizeof(double));
    col_sums = (double*)calloc(n, sizeof(double));

    build_checksum(A, B, C);
	err = verify_checksum(C);
	if ( err < 0 ) {
		printf("err %d\n", err);
	}
    lambda = infnorm(A) * infnorm(B) * k * u;
//	lambda = 1.0;
    mu = onenorm(A) * onenorm(B) * k * u;
    tau = (lambda < mu) ? lambda : mu;
//	lambda = mu = tau = 1.e-3;
	printf("infnorm(A), onenorm(A), infnorm(B),onenorm(B):%g, %g, %g, %g\n",
			infnorm(A), onenorm(A), infnorm(B),onenorm(B) );
	printf("lambda,mu,tau:%g %g %g\n", lambda, mu, tau);

//    pthread_t tid;
    //pthread_create(&tid, NULL, noise_thread, NULL);


  
   gettimeofday(&tvs,NULL);
    ft_dgemm(A, B, C, R);
   gettimeofday(&tve,NULL);
   span =  tve.tv_sec-tvs.tv_sec + (tve.tv_usec-tvs.tv_usec)/1000000.0;
    printf("ft_dgemm: %.12fGFLOPS %.12fs\n", 
            2* pow((N/1000.), 3) * (1/span), span);
    //pthread_cancel(tid);

    gettimeofday(&tvs,NULL);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, D);
    gettimeofday(&tve,NULL);
    span =  tve.tv_sec-tvs.tv_sec + (tve.tv_usec-tvs.tv_usec)/1000000.0;
    printf("dgemm: %.12fGFLPS %.12fs\n", 
            2* pow((N/1000.), 3) * (1/span), span);
    gsl_matrix_sub(C,D); 
    printf("error(infnorm): %f\n", infnorm(C));

   

    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(C);
    gsl_matrix_free(D);
    free(row_sums);free(col_sums);
}

int main(int argc, char *argv[])
{
    if (argc < 3){ 
       	printf("Usage: %s matrix_size rank\n", argv[0]);
	printf("It computes matrix mulitiplication via several rank-k update\n");
	printf("the choice of rank determines the check frequency and performance.\n");
        exit(1);
    }

    N = atoi(argv[1]);
    R = atoi(argv[2]);
    test_ft_dgemm();
}