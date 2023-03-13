package ru.alexander1248.thalos.learning;

public class ADAM extends LearningRule {
    private double[][] acceleration;
    private double[][] gradient;

    private final double beta1;
    private final double beta2;

    public ADAM(double learningSpeed, double beta1, double beta2) {
        super(learningSpeed);
        this.beta1 = beta1;
        this.beta2 = beta2;
    }
    private int counter;

    @Override
    public void initialize(int layerSize, int inputSize) {
        acceleration = new double[layerSize][inputSize];
        gradient = new double[layerSize][inputSize];
        counter = 0;
    }

    @Override
    public void calculate(double[][] weights, double[] bias, double[] error, double[] input) {
        double ls = getLearningSpeed();
        double b1Rev = 1 - beta1;
        double b2Rev = 1 - beta2;
        if (counter < 100) {
            for (int i = 0; i < error.length; i++) {
                for (int j = 0; j < input.length; j++) {
                    double shift = error[i] * input[j];
                    acceleration[i][j] *= beta1;
                    acceleration[i][j] += b1Rev * shift;

                    gradient[i][j] *= beta2;
                    gradient[i][j] += b2Rev * shift * shift;

                    double edacc = acceleration[i][j] / (1 - Math.pow(beta1, counter));
                    double edgrad = gradient[i][j] / (1 - Math.pow(beta1, counter));

                    weights[i][j] += ls * edacc / Math.sqrt(edgrad + 1e-8);
                }
                bias[i] += error[i] * ls;
            }
            counter++;
        }
        else {
            for (int i = 0; i < error.length; i++) {
                for (int j = 0; j < input.length; j++) {
                    double shift = error[i] * input[j];
                    acceleration[i][j] *= beta1;
                    acceleration[i][j] += b1Rev * shift;

                    gradient[i][j] *= beta2;
                    gradient[i][j] += b2Rev * shift * shift;

                    weights[i][j] += ls * acceleration[i][j] / Math.sqrt(gradient[i][j] + 1e-8);
                }
                bias[i] += error[i] * ls;
            }
        }
    }

    public static class Builder {
        private double beta1 = 0.9;
        private double beta2 = 0.999;

        private double learningSpeed = 0.1;

        public void setBeta1(double beta1) {
            this.beta1 = beta1;
        }

        public void setBeta2(double beta2) {
            this.beta2 = beta2;
        }

        public void setLearningSpeed(double learningSpeed) {
            this.learningSpeed = learningSpeed;
        }

        public ADAM build() {
            return new ADAM(learningSpeed, beta1, beta2);
        }
    }
}
