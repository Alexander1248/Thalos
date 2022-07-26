package ru.alexander.nnlib;

import com.aparapi.Kernel;
import ru.alexander.nnlib.tools.Matrix;
import ru.alexander.nnlib.tools.ThreadSplitter;
import ru.alexander.nnlib.types.ThreadingType;

public class Layer {
    private float[] input;
    private Matrix weights;
    private final float[] weightedSum;
    private final float[] output;

    private ThreadingType type;

    private ThreadSplitter cpu;
    private Kernel gpu;

    public Layer(int inputSize, int outputSize, ThreadingType type) {
        input = new float[inputSize];
        weightedSum = new float[outputSize];
        output = new float[outputSize];

        weights = new Matrix(inputSize, outputSize);
        for (int x = 0; x < inputSize; x++)
            for (int y = 0; y < outputSize; y++)
                weights.setCell(x, y, (float) (Math.random() * 2 - 1));

        setThreadingType(type);
    }

    public void setThreadingType(ThreadingType type) {
        this.type = type;
        switch (type) {
            case CPU -> {
            }
            case GPU -> {
            }
        }
    }

    public void calculate() {
        switch (type) {
            case CPU -> {
            }
            case GPU -> {
            }
        }
    }

    public void setInput(float[] input) {
        this.input = input;
    }

    public float[] getWeightedSum() {
        return weightedSum;
    }

    public float[] getOutput() {
        return output;
    }
}
