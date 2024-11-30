package org.ulpgc.bigdata.task3;

import java.util.Random;

public class MatrixInitializer {

    private final Random random;

    public MatrixInitializer(long seed) {
        this.random = new Random(seed);
    }

    public float[][] initializeRandomMatrix(int rows, int columns) {
        float[][] matrix = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = random.nextFloat();
            }
        }
        return matrix;
    }
}
