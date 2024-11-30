package org.ulpgc.bigdata.task3;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MatrixMultiplier {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    //Sequential matrix multiplication
    public void sequentialMultiplication(float[][] A, float[][] B, float[][] C) {
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
    public void parallelMultiplication(float[][] A, float[][] B, float[][] C) {
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
    public static void vectorizedSIMDMultiplication(float[][] A, float[][] B, float[][] C) {
        int rows = A.length;
        int cols = B[0].length;
        int commonDim = A[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                FloatVector result = FloatVector.zero(SPECIES);
                float sum = 0;
                int k = 0;
                for (; k + SPECIES.length() <= commonDim; k += SPECIES.length()) {
                    FloatVector a = FloatVector.fromArray(SPECIES, A[i], k);
                    FloatVector b = FloatVector.fromArray(SPECIES, B[j], k);
                    result = result.add(a.mul(b));

                    for (int l = 0; l < SPECIES.length(); l++) {
                        sum += result.lane(l);
                    }
                }

                for (; k < commonDim; k++) {
                    sum += A[i][k] * B[k][j];
                }

                C[i][j] = sum;
            }
        }
    }

}
