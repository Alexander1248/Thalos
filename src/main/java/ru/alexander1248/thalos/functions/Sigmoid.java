package ru.alexander1248.thalos.functions;


public class Sigmoid implements ActivationFunction {
    @Override
    public double function(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        double v = Math.exp(-x);
        return v / Math.pow(1 + v, 2);
    }
}
