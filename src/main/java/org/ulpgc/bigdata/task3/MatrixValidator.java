package org.ulpgc.bigdata.task3;

public class MatrixValidator{

    public boolean validateResults(float[][] C1, float[][] C2) {
        int rows = C1.length;
        int cols = C1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(C1[i][j] - C2[i][j]) > 0.001) {
                    return false;
                }
            }
        }
        return true;
    }
}
