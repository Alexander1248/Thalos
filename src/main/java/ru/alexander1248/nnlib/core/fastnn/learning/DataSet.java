package ru.alexander1248.nnlib.core.fastnn.learning;

import java.util.ArrayList;
import java.util.Arrays;
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

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (DataSetRow row : rows) builder.append(row).append("\n");
        return builder.toString();
    }

    public static class DataSetRow {
        public float[] input;
        public float[] output;

        public DataSetRow(float[] input, float[] output) {
            this.input = input;
            this.output = output;
        }

        @Override
        public String toString() {
            return Arrays.toString(input) + " - " + Arrays.toString(output);
        }
    }
}
