package ru.alexander1248.thalos.learning;

public class RMSProp extends LearningRule {
    private double[][] gradient;

    private final double attenuation;

    public RMSProp(double learningSpeed, double attenuation) {
        super(learningSpeed);
        this.attenuation = attenuation;
    }

    @Override
    public void initialize(int layerSize, int inputSize) {
        gradient = new double[layerSize][inputSize];
    }

    @Override
    public void calculate(double[][] weights, double[] bias, double[] error, double[] input) {
        double ls = getLearningSpeed();
        double aRev = 1 - attenuation;
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < input.length; j++) {
                double shift = error[i] * input[j];
                gradient[i][j] *= attenuation;
                gradient[i][j] += aRev * shift * shift;
                weights[i][j] += shift * ls / Math.sqrt(gradient[i][j] + 1e-7);
            }
            bias[i] += error[i] * ls;
        }
    }
}
