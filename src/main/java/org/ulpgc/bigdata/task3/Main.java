package org.ulpgc.bigdata.task3;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;

public class Main {

    public static void main(String[] args) {
        int n = 3000    ;
        long seed = 42;

        System.out.println("=".repeat(60));
        System.out.printf("%-20s %s %d x %d\n", "Matrix Multiplication", "Size:", n, n);
        System.out.println("=".repeat(60));

        // Initialize matrices
        MatrixInitializer matrixInitializer = new MatrixInitializer(seed);
        float[][] A = matrixInitializer.initializeRandomMatrix(n, n);
        float[][] B = matrixInitializer.initializeRandomMatrix(n, n);
        float[][] C_Sequential = new float[n][n];
        float[][] C_Parallel = new float[n][n];
        float[][] C_Vectorized = new float[n][n];

        MatrixMultiplier matrixMultiplier = new MatrixMultiplier();

        // Sequential multiplication
        System.out.printf("%-30s", "Starting Sequential Multiplication...");
        long start = System.currentTimeMillis();
        matrixMultiplier.sequentialMultiplication(A, B, C_Sequential);
        long end = System.currentTimeMillis();
        long sequentialTime = (end - start) / 1000;
        System.out.printf("Done in: %5d seconds\n", sequentialTime);

        // Parallel multiplication
        System.out.printf("%-30s", "Starting Parallel Multiplication...");
        start = System.currentTimeMillis();
        matrixMultiplier.parallelMultiplication(A, B, C_Parallel);
        end = System.currentTimeMillis();
        long parallelTime = (end - start) / 1000;
        System.out.printf("Done in: %5d seconds\n", parallelTime);

        // Vectorized multiplication
        System.out.printf("%-30s", "Starting Vectorized Multiplication...");
        start = System.currentTimeMillis();
        MatrixMultiplier.vectorizedSIMDMultiplication(A, B, C_Vectorized);
        end = System.currentTimeMillis();
        long vectorizedTime = (end - start) / 1000  ;
        System.out.printf("Done in: %5d seconds\n", vectorizedTime);

        // Validate results
        System.out.println("-".repeat(60));
        MatrixValidator matrixValidator = new MatrixValidator();
        boolean parallelEqual = matrixValidator.validateResults(C_Sequential, C_Parallel);
        boolean vectorizedValidation = matrixValidator.validateResults(C_Sequential, C_Vectorized);

        System.out.printf("%-30s: %s\n", "Is Parallel result correct?", parallelEqual ? "Yes" : "No");
        System.out.printf("%-30s: %s\n", "Is Vectorized result correct?", vectorizedValidation ? "Yes" : "No");

        // Metrics
        System.out.println("-".repeat(60));
        double speedupParallel = (double) sequentialTime / parallelTime;
        double speedupVectorized = (double) sequentialTime / vectorizedTime;
        System.out.printf("%-30s: %.2fx\n", "Speedup Parallel", speedupParallel);
        System.out.printf("%-30s: %.2fx\n", "Speedup Vectorized", speedupVectorized);

        // Resource usage
        logResourceUsage();
        System.out.println("=".repeat(60));
    }

    private static void logResourceUsage() {
        System.out.printf("\n%-30s\n", "Final Resource Usage");
        System.out.println("-".repeat(60));

        int numCores = Runtime.getRuntime().availableProcessors();
        System.out.printf("%-30s: %d\n", "Number of Cores:", numCores);

        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heapMemoryUsage = memoryBean.getHeapMemoryUsage();
        System.out.printf("%-30s: %d MB\n", "Initial Heap Memory:", heapMemoryUsage.getInit() / 1024 / 1024);
        System.out.printf("%-30s: %d MB\n", "Maximum Heap Memory:", heapMemoryUsage.getMax() / 1024 / 1024);
        System.out.printf("%-30s: %d MB\n", "Current Heap Memory:", heapMemoryUsage.getUsed() / 1024 / 1024);

    }

}