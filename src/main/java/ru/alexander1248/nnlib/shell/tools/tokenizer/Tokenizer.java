package ru.alexander1248.nnlib.shell.tools.tokenizer;

import java.util.List;

public interface Tokenizer {
    float[] tokenize(String data);
    String detokenize(float[] data);
}
