package ru.alexander1248.nnlib.core.customnn.learning;

import ru.alexander1248.nnlib.core.customnn.CustomNeuralNetwork;
import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;

public abstract class CustomNNLearningRule {
    protected CustomNeuralNetwork network;

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

    public void setNetwork(CustomNeuralNetwork network) {
        this.network = network;
    }
}
