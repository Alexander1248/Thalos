package ru.alexander1248.nnlib.shell.tools.tokenizer;

import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SimpleValueProximityTokenizer implements Tokenizer {
    private final List<Token> tokens = new ArrayList<>();

    public SimpleValueProximityTokenizer() {
        tokens.add(new Token(" ", new float[Token.tokenKeySize]));
    }

    public boolean addToken(Token token) {
        if (tokens.stream().noneMatch(t -> token.value.equals(t.value))) {
            tokens.add(token);
            return true;
        }
        else return false;
    }

    public Token getTokenByValue(String value) {
        return tokens.stream().filter(t -> t.value.equals(value)).findAny().orElse(null);
    }
    public Token getTokenByKey(float[] key) {
        return tokens.stream().filter(t -> Arrays.equals(t.key, key)).findAny().orElse(null);
    }

    @Override
    public float[] tokenize(String data) {
        if (!data.isEmpty()) {
            String[] s = data.split(" ");
            float[] spaceToken = getTokenByValue(" ").key;
            float[] out = new float[tokens.size() * Token.tokenKeySize];
            int i = 0;

            float[] buff = getTokenByValue(s[0]).key;
            for (int j = 0; j < buff.length; j++) {
                out[i] = buff[j];
                i++;
            }

            for (int t = 1; t < s.length; t++) {
                for (int j = 0; j < spaceToken.length; j++) {
                    out[i] = spaceToken[j];
                    i++;
                }

                buff = getTokenByValue(s[i]).key;
                for (int j = 0; j < buff.length; j++) {
                    out[i] = buff[j];
                    i++;
                }
            }

            return out;
        }
        return null;
    }

    @Override
    public String detokenize(float[] data) {
        if (data.length >= Token.tokenKeySize) {
            StringBuilder out = new StringBuilder();
            float[] buff = new float[Token.tokenKeySize];

            for (int i = 0; i < data.length; i += Token.tokenKeySize) {
                for (int j = 0; j < Token.tokenKeySize; j++) buff[j] = data[i + j];
                out.append(getTokenByKey(buff).value);
            }
            return out.toString();
        }
        return null;
    }
}
