package ru.alexander1248.nnlib.shell.tools.tokenizer;

import java.util.List;

public interface Tokenizer {
    List<float[]> tokenize(String data);
    String detokenize(List<float[]> data);
}
