package org.ulpgc.bigdata.task3;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MatrixMultiplier {

    //Sequential matrix multiplication
    public void sequentialMultiplication(int[][] A, int[][] B, int[][] C) {
        int rows = A.length;
        int columns = B[0].length;
        int common = A[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                for (int k = 0; k < common; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    //Parallel matrix multiplication
    public void parallelMultiplication(int[][] A, int[][] B, int[][] C) {
        int rows = A.length;
        int columns = B[0].length;
        int common = A[0].length;

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (int i = 0; i < rows; i++) {
            int row = i;
            executor.execute(() -> {
                for (int j = 0; j < columns; j++) {
                    C[row][j] = 0;
                    for (int k = 0; k < common; k++) {
                        C[row][j] += A[row][k] * B[k][j];
                    }
                }
            });
        }

        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                System.err.println("Parallel execution timeout has expired.");
            }
        } catch (InterruptedException e) {
            System.err.println("Parallel execution was interrupted.");
            Thread.currentThread().interrupt();
        }
    }

    //Vectorized multiplication
    public INDArray vectorMultiplication(INDArray A, INDArray B) {
        return A.mmul(B);
    }

    //Method to convert a 2D array to an INDArray
    public INDArray toINDArray(int[][] matrix) {
        return Nd4j.create(matrix);
    }

    //Method to convert an INDArray to a 2D array
    public int[][] toArray(INDArray matrix) {
        int rows = matrix.rows();
        int columns = matrix.columns();
        int[][] result = new int[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = matrix.getInt(i, j);
            }
        }
        return result;
    }
}
