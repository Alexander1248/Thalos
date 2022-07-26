package ru.alexander.nnlib.learning;

import ru.alexander.nnlib.NeuralNetwork;
import ru.alexander.nnlib.tools.DataSet;

public abstract class LearningRule {
    protected NeuralNetwork network;
    private float learningSpeed = 0.1f;

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
