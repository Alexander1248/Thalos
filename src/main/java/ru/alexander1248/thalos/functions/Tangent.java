package ru.alexander1248.thalos.functions;


public class Tangent implements ActivationFunction {
    @Override
    public double function(double x) {
        return 2.0 / (1 + Math.exp(-x)) - 1;
    }

    @Override
    public double derivative(double x) {
        double v = Math.exp(-x);
        return 2 * v / Math.pow(1 + v, 2);
    }
}
