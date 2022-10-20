package ru.alexander1248.nnlib.core.fastnn.learning;

import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public abstract class LearningRule {
    protected NeuralNetwork network;
    protected ThreadingType workingType;

    protected long maxIterations = -1;
    protected long iteration = 0;

    public void setMaxIterations(long maxIterations) {
        this.maxIterations = maxIterations;
    }
    public long getMaxIterations() {
        return maxIterations;
    }
    public long getIteration() {
        return iteration;
    }


    public abstract void learn();

    public void setNetwork(NeuralNetwork network) {
        this.network = network;
    }
}
