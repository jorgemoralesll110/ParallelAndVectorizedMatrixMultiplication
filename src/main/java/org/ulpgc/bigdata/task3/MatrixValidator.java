package org.ulpgc.bigdata.task3;

public class MatrixValidator{

    public boolean validateResults(int[][] C1, int[][] C2) {
        int rows = C1.length;
        int cols = C1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (C1[i][j] != C2[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
}
