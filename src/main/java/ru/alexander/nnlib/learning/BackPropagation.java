package ru.alexander.nnlib.learning;

import ru.alexander.nnlib.Layer;
import ru.alexander.nnlib.exceptions.EmptyNeuralNetworkException;
import ru.alexander.nnlib.exceptions.NoInputLayerException;
import ru.alexander.nnlib.tools.DataSet;

import java.util.List;

public class BackPropagation extends LearningRule {
    private float maxError = 0.1f;

    private float[][] error;

    @Override
    public void learn(DataSet dataSet) {
        try {
            List<Layer> layers = network.getLayers();
            error = new float[layers.size()][];
            float err;
            do {
                err = 0;
                for (DataSet.DataSetRow row : dataSet.getRows()) {
                    network.setInput(row.input);
                    network.calculate();

                    calculateError(layers, row);
                    calculateWeights(layers, row);
                }
            } while (err > maxError);
        } catch (NoInputLayerException | EmptyNeuralNetworkException e) {
            throw new RuntimeException(e);
        }
    }
    private void calculateError(List<Layer> layers, DataSet.DataSetRow row) {
        Layer outLayer = layers.get(layers.size() - 1);
        float[] output = outLayer.getOutput();
        error[layers.size() - 1] = new float[output.length];
        for (int current = 0; current < output.length; current++) {
            error[layers.size() - 1][current] = (row.output[current] - output[current]);

            float wsum = outLayer.getWeightedSum()[current];
            double pow = Math.pow(1 + Math.exp(-wsum), 2);
            if (outLayer.getAfType() == 1) error[layers.size() - 1][current] *= Math.exp(-wsum) / pow;
            else if (outLayer.getAfType() == 2) error[layers.size() - 1][current] *= 2f * Math.exp(-wsum) / pow;
            else if (outLayer.getAfType() == 3) error[layers.size() - 1][current] *= 1f / (1f + (float)Math.exp(-wsum));
            else if (outLayer.getAfType() == 4) error[layers.size() - 1][current] *= wsum > 0 ? 1 : 0;
            else if (outLayer.getAfType() == 5) error[layers.size() - 1][current] *= wsum > 0 ? 1 : 0.01;
            else if (outLayer.getAfType() == 6) error[layers.size() - 1][current] *= ((wsum + 1) * Math.exp(-wsum) + 1) / pow;
        }

        for (int l = layers.size() - 2; l >= 0; l--) {
            error[l] = new float[output.length];
            Layer layer = layers.get(l);
            for (int current = 0; current < layer.getLayerSize(); current++) {
                float e = 0;
                for (int next = 0; next < layers.get(l + 1).getLayerSize(); next++)
                    e += layers.get(l + 1).getWeights()[current + next * layer.getLayerSize()] * error[l + 1][next];

                error[l][current] = e;

                float wsum = outLayer.getWeightedSum()[current];
                double pow = Math.pow(1 + Math.exp(-wsum), 2);
                if (outLayer.getAfType() == 1) error[l][current] *= Math.exp(-wsum) / pow;
                else if (outLayer.getAfType() == 2) error[l][current] *= 2f * Math.exp(-wsum) / pow;
                else if (outLayer.getAfType() == 3) error[l][current] *= 1f / (1f + (float)Math.exp(-wsum));
                else if (outLayer.getAfType() == 4) error[l][current] *= wsum > 0 ? 1 : 0;
                else if (outLayer.getAfType() == 5) error[l][current] *= wsum > 0 ? 1 : 0.01;
                else if (outLayer.getAfType() == 6) error[l][current] *= ((wsum + 1) * Math.exp(-wsum) + 1) / pow;
            }
        }
    }
    private void calculateWeights(List<Layer> layers, DataSet.DataSetRow row) {
        for (int current = 0; current < layers.get(0).getLayerSize(); current++) {
            for (int prev = 0; prev < row.input.length; prev++) {
                layers.get(0).getWeights()[current * row.input.length + prev] += error[0][current] * row.input[prev] * getLearningSpeed();
            }
            layers.get(0).getBiasWeights()[current] += error[0][current] * getLearningSpeed();
        }

        for (int l = 1; l < layers.size(); l++) {
            for (int i = 0; i < layers.get(l).getLayerSize(); i++) {
                for (int j = 0; j < layers.get(l - 1).getLayerSize(); j++) {
                    layers.get(l).getWeights()[i * row.input.length + j] += error[l][i] * layers.get(l - 1).getOutput()[j] * getLearningSpeed();
                }
                layers.get(l).getBiasWeights()[i] += error[l][i] * getLearningSpeed();
            }
        }
    }

    public void setMaxError(float maxError) {
        this.maxError = maxError;
    }

    public float getMaxError() {
        return maxError;
    }
}
