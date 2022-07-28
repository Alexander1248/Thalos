package ru.alexander1248.nnlib.core.exceptions;

public class EmptyNeuralNetworkException extends Exception {
    public EmptyNeuralNetworkException() {
    }
    public EmptyNeuralNetworkException(String message) {
        super(message);
    }
}
