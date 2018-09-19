#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int THRESHOLD;

int** readMatrix(FILE*, int, int);
int** newMatrix(int, int);
void Strassen_2(int**, int**, int**);
void Strassen(int**, int**, int**, int);
void Matrix_Mult(int**, int**, int**, int);
void generalMult(int**, int**, int**, int);
void transpose(int**, int**, int);
void Matrix_Add(int**, int**, int**, int);
void Matrix_Sub(int**, int**, int**, int);
int getPaddingSize(int);

int main(int argc, char* argv[]) {

	double t1 = omp_get_wtime();
	int M, N, P;
	// read in matrix in file.
	FILE *fp;
	if (fopen_s(&fp, (argc == 1) ? "input.txt" : argv[1], "r")) exit(1);
	fscanf_s(fp, "%d%d", &M, &N);
	if (M >= 2048) THRESHOLD = 1024;
	else THRESHOLD = 128;
	int **matrixA = readMatrix(fp, M, N);
	fscanf_s(fp, "%d%d", &N, &P);
	int **matrixB = readMatrix(fp, N, P);
	
	// strat multiplication.
	omp_set_num_threads(8);
	double t2 = omp_get_wtime();
	int **result = newMatrix(M, P);
	Strassen(result, matrixA, matrixB, getPaddingSize(M));
	double t3 = omp_get_wtime();

	// output result and performance to file.
	if (fopen_s(&fp, "Output_Strassen.txt", "w")) exit(1);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < P; j++) {
			fprintf_s(fp, "%d ", result[i][j]);
		}
		fprintf_s(fp, "\r\n");
	}
	double t4 = omp_get_wtime();

	fclose(fp);
	FILE *performance_output;
	if (fopen_s(&performance_output, "Performance_Strassen.txt", "w")) exit(1);
	fprintf_s(performance_output, "Total: %.3f msec\n", (t4 - t1) * 1000);
	fprintf_s(performance_output, "ReadF: %.3f msec\n", (t2 - t1) * 1000);
	fprintf_s(performance_output, "Compu: %.3f msec (Compute only)\n", (t3 - t2) * 1000);
	fprintf_s(performance_output, "Compu: %.3f msec (I/O only)\n", (t4 - t3) * 1000);
	fprintf_s(performance_output, "Compu: %.3f msec (All part)\n", (t4 - t2) * 1000);
	fclose(performance_output);

	free(matrixA);
	free(matrixB);
	free(result);
	return 0;
}

int** readMatrix(FILE *fp, int col, int row) {
	int** matrix = newMatrix(col, row);
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			if (!fscanf_s(fp, "%d", &matrix[i][j])) break;
		}
	}
	return matrix;
}

int** newMatrix(int col, int row) {
	col = getPaddingSize(col);
	row = getPaddingSize(row);
	int** matrix = (int**)malloc(col * sizeof(int*));
	for (int i = 0; i < row; i++)
		matrix[i] = (int*)calloc(row, sizeof(int));
	return matrix;
}

void Strassen_2(int **result, int **A, int **B) {
	int p1 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]);
	int p2 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]);
	int p3 = (A[0][0] - A[1][0]) * (B[0][0] + B[0][1]);
	int p4 = (A[0][0] + A[0][1]) * B[1][1];
	int p5 = A[0][0] * (B[0][1] - B[1][1]);
	int p6 = A[1][1] * (B[1][0] - B[0][0]);
	int p7 = (A[1][0] + A[1][1]) * B[0][0];

	result[0][0] = p1 + p2 - p4 + p6;
	result[0][1] = p4 + p5;
	result[1][0] = p6 + p7;
	result[1][1] = p2 - p3 + p5 - p7;
}

void Strassen(int **result, int **A, int **B, int n)
{
	// if(n == 2) Strassen_2(result, A, B);
	if (n <= THRESHOLD) generalMult(result, A, B, n);
	else
	{
		int i, j, n2 = n / 2;
		int **p1 = newMatrix(n2, n2);
		int **p2 = newMatrix(n2, n2);
		int **p3 = newMatrix(n2, n2);
		int **p4 = newMatrix(n2, n2);
		int **p5 = newMatrix(n2, n2);
		int **p6 = newMatrix(n2, n2);
		int **p7 = newMatrix(n2, n2);

		int **A11 = newMatrix(n2, n2);
		int **A12 = newMatrix(n2, n2);
		int **A21 = newMatrix(n2, n2);
		int **A22 = newMatrix(n2, n2);
		int **B11 = newMatrix(n2, n2);
		int **B12 = newMatrix(n2, n2);
		int **B21 = newMatrix(n2, n2);
		int **B22 = newMatrix(n2, n2);
		int **result11 = newMatrix(n2, n2);
		int **result12 = newMatrix(n2, n2);
		int **result21 = newMatrix(n2, n2);
		int **result22 = newMatrix(n2, n2);

		int **Arst1 = newMatrix(n2, n2);
		int **Brst1 = newMatrix(n2, n2);
		int **Arst2 = newMatrix(n2, n2);
		int **Brst2 = newMatrix(n2, n2);
		int **Arst3 = newMatrix(n2, n2);
		int **Brst3 = newMatrix(n2, n2);
		int **Arst4 = newMatrix(n2, n2);
		int **Brst5 = newMatrix(n2, n2);
		int **Brst6 = newMatrix(n2, n2);
		int **Arst7 = newMatrix(n2, n2);

#pragma omp parallel for
		for (i = 0; i < n2; ++i) {
			A11[i] = &A[i][0];
			A12[i] = &A[i][n2];
			A21[i] = &A[i + n2][0];
			A22[i] = &A[i + n2][n2];

			B11[i] = &B[i][0];
			B12[i] = &B[i][n2];
			B21[i] = &B[i + n2][0];
			B22[i] = &B[i + n2][n2];

			result11[i] = &result[i][0];
			result12[i] = &result[i][n2];
			result21[i] = &result[i + n2][0];
			result22[i] = &result[i + n2][n2];
		}
#pragma omp parallel sections
		{
#pragma omp section
			{
				// p1 = (a12 - a22) * (b21 + b22)
				Matrix_Sub(Arst1, A12, A22, n2);
				Matrix_Add(Brst1, B21, B22, n2);
				Strassen(p1, Arst1, Brst1, n2);
			}

#pragma omp section
			{
				// p2 = (a11 + a22) * (b11 + b22)
				Matrix_Add(Arst2, A11, A22, n2);
				Matrix_Add(Brst2, B11, B22, n2);
				Strassen(p2, Arst2, Brst2, n2);
			}

#pragma omp section
			{
				// p3 = (a11 - a21) * (b11 + b12)
				Matrix_Sub(Arst3, A11, A21, n2);
				Matrix_Add(Brst3, B11, B12, n2);
				Strassen(p3, Arst3, Brst3, n2);
			}

#pragma omp section 
			{
				// p4 = (a11 + a12) * b22
				Matrix_Add(Arst4, A11, A12, n2);
				Strassen(p4, Arst4, B22, n2);
			}

#pragma omp section
			{
				// p5 = a11 * (b12 - b22)
				Matrix_Sub(Brst5, B12, B22, n2);
				Strassen(p5, A11, Brst5, n2);
			}

#pragma omp section
			{
				// p6 = a22 * (b21 - b11)
				Matrix_Sub(Brst6, B21, B11, n2);
				Strassen(p6, A22, Brst6, n2);
			}

#pragma omp section
			{
				// p7 = (a21 + a22) * b11
				Matrix_Add(Arst7, A21, A22, n2);
				Strassen(p7, Arst7, B11, n2);
			}
		}
// #pragma omp barrier

#pragma omp parallel for private(j)
		for (i = 0; i < n2; ++i) {
			for (j = 0; j < n2; ++j) {
				result11[i][j] = p1[i][j] + p2[i][j] - p4[i][j] + p6[i][j];
				result12[i][j] = p4[i][j] + p5[i][j];
				result21[i][j] = p6[i][j] + p7[i][j];
				result22[i][j] = p2[i][j] - p3[i][j] + p5[i][j] - p7[i][j];
			}
		}

		free(p1); free(p2); free(p3); free(p4);
		free(p5); free(p6); free(p7);
		free(A11); free(A12); free(A21); free(A22);
		free(B11); free(B12); free(B21); free(B22);
		free(result11); free(result12); free(result21); free(result22);
		free(Arst1); free(Brst1); free(Arst2);
		free(Brst2); free(Arst3); free(Brst3);
		free(Arst4); free(Brst5); free(Brst6); free(Arst7);
	}
}

void Matrix_Mult(int **result, int **A, int **B, int n) {
#pragma omp parallel 
	{
#pragma omp single
		{
			Strassen(result, A, B, n);
		}
	}
}

void Matrix_Add(int** result, int** A, int** B, int n) {
	int i, j, temp = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			result[i][j] = A[i][j] + B[i][j];
		}
	}
}

void Matrix_Sub(int** result, int** A, int** B, int n) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			result[i][j] = A[i][j] - B[i][j];
		}
	}
}

void generalMult(int** result, int** A, int** B, int n) {
	int rst;
	for (int k = 0; k < n; k++) {
		for (int i = 0; i < n; i++) {
			rst = A[i][k];
			for (int j = 0; j < n; j++)
				result[i][j] += rst * B[k][j];
		}
	}
}

void transpose(int **A, int **B, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			B[i][j] = A[j][i];
		}
	}
}

// get the smallest padding m*2^n that m < THRESHOLD
int getPaddingSize(int size) {
	int cnt = 0;
	int n = size;
	while (n > THRESHOLD) {
		cnt++;
		n /= 2;
	}
	if (size % (1 << cnt) == 0) {
		return size;
	}
	else {
		return size + (1 << cnt) - size % (1 << cnt);
	}
}