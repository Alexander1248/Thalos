package ru.alexander1248.thalos.learning;

public class AdaGrad extends LearningRule {
    private double[][] gradient;


    public AdaGrad(double learningSpeed) {
        super(learningSpeed);
    }

    @Override
    public void initialize(int layerSize, int inputSize) {
        gradient = new double[layerSize][inputSize];
    }

    @Override
    public void calculate(double[][] weights, double[] bias, double[] error, double[] input) {
        double ls = getLearningSpeed();
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < input.length; j++) {
                double shift = error[i] * input[j];
                gradient[i][j] += shift * shift;
                weights[i][j] += shift * ls / Math.sqrt(gradient[i][j] + 1e-7);
            }
            bias[i] += error[i] * ls;
        }
    }
}
