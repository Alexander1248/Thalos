package ru.alexander.nnlib.exceptions;

public class EmptyNeuralNetworkException extends Exception {
    public EmptyNeuralNetworkException() {
    }
    public EmptyNeuralNetworkException(String message) {
        super(message);
    }
}
