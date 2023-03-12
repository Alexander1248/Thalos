package ru.alexander1248.thalos.learning;

public class Nesterov extends LearningRule {
    private double[][] acceleration;

    private final double momentum;

    public Nesterov(double learningSpeed, double momentum) {
        super(learningSpeed);
        this.momentum = momentum;
    }

    @Override
    public void initialize(int layerSize, int inputSize) {
        acceleration = new double[layerSize][inputSize];
    }

    @Override
    public void calculate(double[][] weights, double[] bias, double[] error, double[] input) {
        double ls = getLearningSpeed();
        double mRev = 1 - momentum;
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < input.length; j++) {
                acceleration[i][j] *= momentum;
                acceleration[i][j] += mRev * error[i] * input[j] * ls;
                weights[i][j] += acceleration[i][j];
            }
            bias[i] += error[i] * ls;
        }
    }
}
