package ru.alexander1248.thalos.exceptions;

public class EmptyNeuralNetworkException extends RuntimeException {
    public EmptyNeuralNetworkException() {
    }
    public EmptyNeuralNetworkException(String message) {
        super(message);
    }
}