package ru.alexander1248.nnlib.shell.tools.tokenizer;

public class Token {
    public static int tokenKeySize = 32;
    public String value;
    public float[] key;

    public Token(String value, float[] key) {
        if (key.length == tokenKeySize) {
            this.value = value;
            this.key = key;
        }
    }

    public static Token random(String value) {
        float[] data = new float[tokenKeySize];
        for(int i = 0; i < tokenKeySize; i++) data[i] = (float) Math.random();
        return new Token(value, data);
    }
    public static Token random(Token nearest, String value) {
        float[] data = new float[tokenKeySize];
        for(int i = 0; i < tokenKeySize; i++) {
            if (Math.random() <= 1f / tokenKeySize) data[i] = (float) Math.random();
            else data[i] = nearest.key[i];
        }
        data[(int) (Math.random() * tokenKeySize)] = (float) Math.random();
        return new Token(value, data);
    }
    public static Token add(Token a, Token b, String value) {
        float[] data = new float[tokenKeySize];
        for(int i = 0; i < tokenKeySize; i++) data[i] = a.key[i] + b.key[i];
        return new Token(value, data);
    }
    public static Token sub(Token a, Token b, String value) {
        float[] data = new float[tokenKeySize];
        for(int i = 0; i < tokenKeySize; i++) data[i] = a.key[i] - b.key[i];
        return new Token(value, data);
    }

    public static Token combine(Token base, Token sub, Token add, String value) {
        float[] data = new float[tokenKeySize];
        for(int i = 0; i < tokenKeySize; i++) data[i] = base.key[i] - sub.key[i] + add.key[i];
        return new Token(value, data);
    }
}
