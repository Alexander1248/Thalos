package ru.alexander1248.thalos.learning;

public class Standard extends LearningRule {

    public Standard(double learningSpeed) {
        super(learningSpeed);
    }

    @Override
    public void initialize(int layerSize, int inputSize) {}

    @Override
    public void calculate(double[][] weight, double[] bias, double[] error, double[] input) {
        double ls = getLearningSpeed();
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < input.length; j++)
                weight[i][j] += error[i] * input[j] * ls;
            bias[i] += error[i] * ls;
        }
    }
}
