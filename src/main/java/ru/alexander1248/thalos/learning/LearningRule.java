package ru.alexander1248.thalos.learning;

public abstract class LearningRule {
    private final double learningSpeed;

    public LearningRule(double learningSpeed) {
        this.learningSpeed = learningSpeed;
    }

    protected double getLearningSpeed() {
        return learningSpeed;
    }

    public abstract void initialize(int layerSize, int inputSize);

    public abstract void calculate(double[][] weight, double[] bias, double[] error, double[] input);
}
