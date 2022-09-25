#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <vector>

/*
// Visual Studio
In "Project/Properties/Linker/Input" add "cublas.lib; cusparse.lib;"

// Linux
nvcc -Xcompiler '-fPIC' -lcublas -lcusparse -shared -o libbicgstab.so bicgstab.cu
*/

cusparseHandle_t cusparseHandle;
cublasHandle_t cublasHandle;

/// Used for extra memory use in LU decomposition.
void *pBuffer;

/// Set up descriptor for A.
void setUpDescriptor(cusparseMatDescr_t &descrA, cusparseMatrixType_t matrixType, cusparseIndexBase_t indexBase) {
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, matrixType);
    cusparseSetMatIndexBase(descrA, indexBase);
}

/// Set up descriptor for LU.
void setUpDescriptorLU(cusparseMatDescr_t &descrLU, cusparseMatrixType_t matrixType,
                       cusparseIndexBase_t indexBase, cusparseFillMode_t fillMode,
                       cusparseDiagType_t diagType) {
    cusparseCreateMatDescr(&descrLU);
    cusparseSetMatType(descrLU, matrixType);
    cusparseSetMatIndexBase(descrLU, indexBase);
    cusparseSetMatFillMode(descrLU, fillMode);
    cusparseSetMatDiagType(descrLU, diagType);
}

/// Memory query for LU.
void memoryQueryLU(csrilu02Info_t &infoA, csrsv2Info_t &infoL, csrsv2Info_t &infoU,
                   cusparseHandle_t cusparseHandle, const int n, const int nnz,
                   cusparseMatDescr_t &descrA, cusparseMatDescr_t &descrL, cusparseMatDescr_t &descrU,
                   double *d_A, const int *d_A_RowPtr, const int *d_A_ColInd,
                   cusparseOperation_t matrixOperation, void **pBuffer) {
    cusparseCreateCsrilu02Info(&infoA);
    cusparseCreateCsrsv2Info(&infoL);
    cusparseCreateCsrsv2Info(&infoU);

    int pBufferSize_M, pBufferSize_L, pBufferSize_U;
    cusparseDcsrilu02_bufferSize(cusparseHandle, n, nnz, descrA, d_A, d_A_RowPtr,
                                 d_A_ColInd, infoA, &pBufferSize_M);
    cusparseDcsrsv2_bufferSize(cusparseHandle, matrixOperation, n, nnz, descrL,
                               d_A, d_A_RowPtr, d_A_ColInd, infoL, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(cusparseHandle, matrixOperation, n, nnz, descrU,
                               d_A, d_A_RowPtr, d_A_ColInd, infoU, &pBufferSize_U);

    int pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_U));

    cudaMalloc((void **) pBuffer, pBufferSize);
}

/// Analysis for LU.
void analyzeLU(csrilu02Info_t &infoA, csrsv2Info_t &infoL,
               csrsv2Info_t &infoU, cusparseHandle_t cusparseHandle, const int N,
               const int nnz, cusparseMatDescr_t descrA, cusparseMatDescr_t &descrL,
               cusparseMatDescr_t &descrU, double *d_A, const int *d_A_RowPtr,
               const int *d_A_ColInd, cusparseOperation_t matrixOperation,
               cusparseSolvePolicy_t solvePolicy1, cusparseSolvePolicy_t solvePolicy2,
               void *pBuffer) {
    int structural_zero;

    cusparseDcsrilu02_analysis(cusparseHandle, N, nnz, descrA, d_A, d_A_RowPtr,
                               d_A_ColInd, infoA, solvePolicy1, pBuffer);

    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(cusparseHandle, infoA, &structural_zero);

    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d, %d) is missing\n", structural_zero, structural_zero);
    }

    cusparseDcsrsv2_analysis(cusparseHandle, matrixOperation, N, nnz, descrL,
                             d_A, d_A_RowPtr, d_A_ColInd, infoL, solvePolicy1, pBuffer);
    cusparseDcsrsv2_analysis(cusparseHandle, matrixOperation, N, nnz, descrU,
                             d_A, d_A_RowPtr, d_A_ColInd, infoU, solvePolicy2, pBuffer);
}

/// Incomplete LU decomposition.
void computeLU(csrilu02Info_t &infoA, cusparseHandle_t cusparseHandle,
               const int N, const int nnz, cusparseMatDescr_t &descrA,
               double *d_A, const int *d_A_RowPtr, const int *d_A_ColInd,
               cusparseSolvePolicy_t solutionPolicy, void *pBuffer) {
    int numericalZero;

    cusparseDcsrilu02(cusparseHandle, N, nnz, descrA, d_A, d_A_RowPtr, d_A_ColInd,
                      infoA, solutionPolicy, pBuffer);

    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(cusparseHandle, infoA,
                                                          &numericalZero);

    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("U(%d, %d) is zero\n", numericalZero, numericalZero);
    }
}

void getIncompleteLU(cusparseHandle_t &cusparseHandle, cusparseMatDescr_t &descrA,
                     cusparseMatDescr_t &descrL, cusparseMatDescr_t &descrU, csrilu02Info_t &infoA,
                     csrsv2Info_t &infoL, csrsv2Info_t &infoU, int n, int nnz, double *valACopy,
                     const int *rowPtr, const int *colInd) {
    // Step 1: Set up descriptors for A, L and U.
    setUpDescriptor(descrA, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO);
    setUpDescriptorLU(descrL, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);
    setUpDescriptorLU(descrU, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // Step 2: Query how much memory used in LU factorization and the two following system inversions.
    memoryQueryLU(infoA, infoL, infoU, cusparseHandle, n, nnz, descrA, descrL, descrU,
                  valACopy, rowPtr, colInd, CUSPARSE_OPERATION_NON_TRANSPOSE, &pBuffer);

    // Step 3: Analyze the three problems: LU factorization and the two following system inversions.
    analyzeLU(infoA, infoL, infoU, cusparseHandle, n, nnz, descrA, descrL, descrU,
              valACopy, rowPtr, colInd, CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_SOLVE_POLICY_NO_LEVEL, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    // Step 4: Factorize A = L * U (A will be overwritten).
    computeLU(infoA, cusparseHandle, n, nnz, descrA, valACopy, GRowPtr, GColInd,
              CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
}

void spSolverBiCGStab(int n, int nnz, const double *valA, const int *rowPtr, const int *colInd,
                      const double *b, double *x, double tol) {
    // Create descriptors for A, L and U.
    cusparseMatDescr_t descrA, descrL, descrU;

    // Create ILU and SV info for A, L and U.
    csrilu02Info_t infoA;
    csrsv2Info_t infoL, infoU;

    // Create a copy of A for incomplete LU decomposition.
    // This copy will be modified in the solving process.
    double *valACopy;
    cudaMalloc((void **) &valACopy, nnz * sizeof(double));
    cudaMemcpy(valACopy, valA, nnz * sizeof(double), cudaMemcpyDeviceToDevice);

    // Incomplete LU.
    getIncompleteLU(cusparseHandle, descrA, descrL, descrU, infoA, infoL, infoU, n, nnz, valACopy, rowPtr, colInd);

    double *r;
    cudaMalloc((void **) &r, n * sizeof(double));
    double *rw;
    cudaMalloc((void **) &rw, n * sizeof(double));
    double *p;
    cudaMalloc((void **) &p, n * sizeof(double));
    double *ph;
    cudaMalloc((void **) &ph, n * sizeof(double));
    double *t;
    cudaMalloc((void **) &t, n * sizeof(double));
    double *q;
    cudaMalloc((void **) &q, n * sizeof(double));
    double *s;
    cudaMalloc((void **) &s, n * sizeof(double));

    double one = 1, nega_one = -1, zero = 0;
    double alpha, negalpha, beta, omega, nega_omega;
    double temp1, temp2;
    double rho = 0.0, rhop;
    double nrmr0, nrmr;
    int niter = 0;

    // Initial guess x0 (all zeros here).
    cublasDscal_v2(cublasHandle, n, &zero, x, 1);

    // 1: compute the initial residual r = b - A * x0.
    cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &nega_one, descrA, valA, rowPtr,
                   colInd, x, &zero, r);
    cublasDaxpy_v2(cublasHandle, n, &one, b, 1, r, 1);

    // 2: copy r into rw and p.
    cublasDcopy_v2(cublasHandle, n, r, 1, rw, 1);
    cublasDcopy_v2(cublasHandle, n, r, 1, p, 1);

    //cublasDnrm2_v2(cublasHandle, n, r, 1, &nrmr0);
    //cudaDeviceSynchronize();

    // Repeat until convergence.
    while (true) {
        rhop = rho;
        cublasDdot_v2(cublasHandle, n, rw, 1, r, 1, &rho);

        if (niter > 0) {
            // 12
            beta = (rho / rhop) * (alpha / omega);

            // 13, p = r + beta * (p - omega * v)
            cublasDaxpy_v2(cublasHandle, n, &nega_omega, q, 1, p, 1);  // p += -omega * v
            cublasDscal_v2(cublasHandle, n, &beta, p, 1);  // p *= beta
            cublasDaxpy_v2(cublasHandle, n, &one, r, 1, p, 1);  // p += 1 * r
        }

        // 15: solve M * pw = p for pw.
        cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &one, descrL, valACopy, rowPtr,
                              colInd, infoL, p, t, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
        cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &one, descrU, valACopy, rowPtr,
                              colInd, infoU, t, ph, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

        // 16
        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, valA, rowPtr, colInd,
                       ph, &zero, q);

        // 17
        cublasDdot_v2(cublasHandle, n, rw, 1, q, 1, &temp1);
        cudaDeviceSynchronize();
        alpha = rho / temp1;
        negalpha = -alpha;

        // 18
        cublasDaxpy_v2(cublasHandle, n, &negalpha, q, 1, r, 1);

        // 19
        cublasDaxpy_v2(cublasHandle, n, &alpha, ph, 1, x, 1);

        // 20
        cublasDnrm2_v2(cublasHandle, n, r, 1, &nrmr);
        cudaDeviceSynchronize();
        if (nrmr < tol) break;

        // 23: solve M * sh = r for sh, note that s is sh for now.
        cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &one, descrL, valACopy, rowPtr,
                              colInd, infoL, r, t, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
        cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &one, descrU, valACopy, rowPtr,
                              colInd, infoU, t, s, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

        // 24
        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, valA, rowPtr, colInd,
                       s, &zero, t);

        // 25: omega = np.dot(t, r) / np.dot(t, t).
        cublasDdot_v2(cublasHandle, n, t, 1, r, 1, &temp1);
        cublasDdot_v2(cublasHandle, n, t, 1, t, 1, &temp2);
        cudaDeviceSynchronize();
        omega = temp1 / temp2;
        nega_omega = -omega;

        // 26
        cublasDaxpy_v2(cublasHandle, n, &omega, s, 1, x, 1);
        cublasDaxpy_v2(cublasHandle, n, &nega_omega, t, 1, r, 1);

        cublasDnrm2_v2(cublasHandle, n, r, 1, &nrmr);
        cudaDeviceSynchronize();
        if (nrmr < tol) break;

        niter++;

        //printf("Norm: %f\n", nrmr);
    }

    //printf("Number of iterations: %d\n", niter);

    // Clean up
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrL);
    cusparseDestroyMatDescr(descrU);
    cusparseDestroyCsrilu02Info(infoA);
    cusparseDestroyCsrsv2Info(infoL);
    cusparseDestroyCsrsv2Info(infoU);
    cudaFree(r);
    cudaFree(rw);
    cudaFree(p);
    cudaFree(ph);
    cudaFree(t);
    cudaFree(q);
    cudaFree(s);
    cudaFree(valACopy);
    cudaFree(pBuffer);;
}

extern "C" {
// Add __declspec(dllexport) if using Visual Studio.
__declspec(dllexport) void solve(int *rowPtr, int *colInd, double *csrData,
                                 double *b, double *x, int n, int nnz, double tol) {
    // Allocate GPU memory
    // ------------------------------------------
    // Copy CSR column indices to GPU.
    int *gColInd;
    cudaMalloc((void **) &gColInd, nnz * sizeof(int));
    cudaMemcpy(gColInd, colInd, nnz * sizeof(int), cudaMemcpyHostToDevice);

    // Copy CSR row offsets to GPU.
    int *gRowPtr;
    cudaMalloc((void **) &gRowPtr, (n + 1) * sizeof(int));
    cudaMemcpy(gRowPtr, rowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Copy CSR data array to GPU.
    double *gCsrData;
    cudaMalloc((void **) &gCsrData, nnz * sizeof(double));
    cudaMemcpy(gCsrData, csrData, nnz * sizeof(double), cudaMemcpyHostToDevice);

    // Residual vector.
    double *gB;
    cudaMalloc((void **) &gB, n * sizeof(double));
    cudaMemcpy(gB, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Solution.
    double *gX;
    cudaMalloc((void **) &gX, n * sizeof(double));
    // ------------------------------------------

    // Create CUDA handles.
    cusparseCreate(&cusparseHandle);
    cublasCreate_v2(&cublasHandle);

    // Solve Ax = b for x.
    spSolverBiCGStab(n, nnz, gCsrData, gRowPtr, gColInd, gB, gX, tol);

    // Copy x back to CPU.
    cudaMemcpy(x, gX, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up.
    cusparseDestroy(cusparseHandle);
    cublasDestroy_v2(cublasHandle);

    cudaFree(gX);
    cudaFree(gB);

    cudaFree(gCsrData);
    cudaFree(gColInd);
    cudaFree(gRowPtr);
}
}

int main() {
    // Sparse linear matrix in CSR.
    std::vector<int> rowPtr{0, 2, 4, 6, 10, 13};
    std::vector<int> colInd{0, 4, 1, 3, 2, 3, 1, 2, 3, 4, 0, 3, 4};
    std::vector<double> csrData{1., 0.36494769, 1., 0.36768485, 1., 0.34217041,
                                0.36768485, 0.34217041, 1., 0.61652355, 0.36494769, 0.61652355,
                                2.36724823};

    // Right-Hand vector.
    std::vector<double> b{0.66419739, 0.33993935, 0.31049594, 0.78174978, 0.36146131};

    // Solution to calculate.
    std::vector<double> x{0, 0, 0, 0, 0};

    // Vector size.
    int n = 5;

    // None-Zero count.
    int nnz = 13;

    // Tolerance.
    double tol = 1e-8;

    solve(rowPtr.data(), colInd.data(), csrData.data(), b.data(), x.data(), n, nnz, tol);

    return 0;
}
