package ru.alexander.nnlib.tools;

import java.util.ArrayList;
import java.util.List;

public class DataSet {
    private final List<DataSetRow> rows = new ArrayList<>();

    public void addRow(DataSetRow row) {
        rows.add(row);
    }
    public void addRow(float[] input, float[] output) {
        rows.add(new DataSetRow(input, output));
    }

    public List<DataSetRow> getRows() {
        return rows;
    }
    public static class DataSetRow {
        public float[] input;
        public float[] output;

        public DataSetRow(float[] input, float[] output) {
            this.input = input;
            this.output = output;
        }
    }
}
