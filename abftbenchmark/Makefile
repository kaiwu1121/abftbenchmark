#BLAS=-framework Accelerate
#BLAS=-L/Users/pwu/Downloads/ATLAS/build/lib -lf77blas -lcblas -latlas
BLAS=-L/root/setupdoc/ATLAS/mybuild/lib -lf77blas -lcblas -latlas
LIBS=-lgsl $(BLAS) -lm
CC=gcc
FLAGS=

%.o:%.c
	$(CC) $(FLAGS) -c $^

ftdgemm_test: ftdgemm_test.o ftdgemm.o ftblas.h
	$(CC) $(FLAGS) $^ $(LIBS) -o $@
	
clean:
		rm *_test *.o

