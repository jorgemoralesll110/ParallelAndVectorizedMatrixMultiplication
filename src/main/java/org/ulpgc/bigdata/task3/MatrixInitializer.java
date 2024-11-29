package org.ulpgc.bigdata.task3;

import java.util.Random;

public class MatrixInitializer {

    private final Random random;

    public MatrixInitializer(long seed) {
        this.random = new Random(seed);
    }

    public int[][] initializeRandomMatrix(int rows, int columns) {
        int[][] matrix = new int[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = random.nextInt();
            }
        }
        return matrix;
    }
}
