package org.ulpgc.bigdata.task3;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;

public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        int n = 2000;
        long seed = 42;

        logger.info("Initializing matrices of size {}x{}\n", n, n);

        // Initialize matrices
        MatrixInitializer matrixInitializer = new MatrixInitializer(seed);
        int[][] A = matrixInitializer.initializeRandomMatrix(n, n);
        int[][] B = matrixInitializer.initializeRandomMatrix(n, n);
        int[][] C_sequential = new int[n][n];
        int[][] C_parallel = new int[n][n];

        MatrixMultiplier matrixMultiplier = new MatrixMultiplier();

        // Sequential multiplication
        logResourceUsage("Before sequential multiplication");
        long start = System.currentTimeMillis();
        logger.debug("Starting sequential multiplication...");
        matrixMultiplier.sequentialMultiplication(A, B, C_sequential);
        long end = System.currentTimeMillis();
        long sequentialTime = end - start;
        logger.info("Sequential multiplication took {} ms", sequentialTime);
        logResourceUsage("After sequential multiplication");

        // Parallel multiplication
        logResourceUsage("Before parallel multiplication");
        start = System.currentTimeMillis();
        logger.debug("Starting parallel multiplication...");
        matrixMultiplier.parallelMultiplication(A, B, C_parallel);
        end = System.currentTimeMillis();
        long parallelTime = end - start;
        logger.info("Parallel multiplication took {} ms", parallelTime);
        logResourceUsage("After parallel multiplication");

        // Vectorized multiplication
        logResourceUsage("Before vector multiplication");
        INDArray ndA = Nd4j.create(A);
        INDArray ndB = Nd4j.create(B);
        start = System.currentTimeMillis();
        logger.debug("Starting vector multiplication...");
        INDArray ndC = matrixMultiplier.vectorMultiplication(ndA, ndB);
        end = System.currentTimeMillis();
        long vectorTime = end - start;
        logger.info("Vector multiplication took {} ms", vectorTime);
        logResourceUsage("After vector multiplication");

        // Compare results
        int[][] C_vectorized = matrixMultiplier.toArray(ndC);

        // Metrics
        double speedupParallel = (double) sequentialTime / parallelTime;
        double speedupVector = (double) sequentialTime / vectorTime;
        int numCores = Runtime.getRuntime().availableProcessors();

        logger.info("Speedup parallel: {}", speedupParallel);
        logger.info("Speedup vector: {}", speedupVector);
        logger.info("Efficiency parallel: {}", speedupParallel / numCores);
        logger.info("Efficiency vector: {}", speedupVector / numCores);

        // Validate results
        MatrixValidator matrixValidator = new MatrixValidator();
        boolean isValidParallel = matrixValidator.validateResults(C_sequential, C_parallel);
        boolean isValidVectorized = matrixValidator.validateResults(C_sequential, C_vectorized);

        logger.info("Parallel multiplication is valid?: {}", isValidParallel);
       logger.info("Vector multiplication is valid?: {}", isValidVectorized);
    }

    // Method to log memory usage
    private static void logResourceUsage(String stage) {
        logger.info("-------------{}-------------", stage);
        int numCores = Runtime.getRuntime().availableProcessors();
        logger.info("Number of cores: {}", numCores);

        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heapMemoryUsage = memoryBean.getHeapMemoryUsage();
        logger.info("Initial memory usage (Heap): {} MB", heapMemoryUsage.getInit() / 1024 / 1024);
        logger.info("Maximum memory usage (Heap): {} MB", heapMemoryUsage.getMax() / 1024 / 1024);
        logger.info("Current Used memory (Heap): {} MB", heapMemoryUsage.getUsed() / 1024 / 1024);
        logger.info("---------------------------------");
    }
}