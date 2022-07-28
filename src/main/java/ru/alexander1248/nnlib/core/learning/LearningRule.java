package ru.alexander1248.nnlib.core.learning;

import ru.alexander1248.nnlib.core.NeuralNetwork;
import ru.alexander1248.nnlib.core.DataSet;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public abstract class LearningRule {
    protected NeuralNetwork network;
    protected ThreadingType workingType;
    private float learningSpeed = 0.1f;

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

    public float getLearningSpeed() {
        return learningSpeed;
    }

    public void setLearningSpeed(float learningSpeed) {
        this.learningSpeed = learningSpeed;
    }

    public abstract void learn(DataSet dataSet);

    public void setNetwork(NeuralNetwork network) {
        this.network = network;
    }
}
