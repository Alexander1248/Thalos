package ru.alexander.nnlib.learning;

import ru.alexander.nnlib.Layer;
import ru.alexander.nnlib.exceptions.EmptyNeuralNetworkException;
import ru.alexander.nnlib.exceptions.NoInputLayerException;
import ru.alexander.nnlib.DataSet;

import java.util.List;

public class MomentumBackPropagation extends BackPropagation {

    private float momentum = 0;
    private float[][][] acceleration;

    @Override
    public void learn(DataSet dataSet) {
        try {
            List<Layer> layers = network.getLayers();
            error = new float[layers.size()][];
            acceleration = new float[layers.size()][][];
            acceleration[0] = new float[layers.get(0).getLayerSize()][network.getInputSize()];
            for (int l = 1; l < layers.size(); l++)
                acceleration[l] = new float[layers.get(l).getLayerSize()][layers.get(l - 1).getLayerSize()];

            do {
                float err = 0;
                for (DataSet.DataSetRow row : dataSet.getRows()) {
                    network.setInput(row.input);
                    network.calculate();

                    err += calculateError(layers, row);
                    calculateWeights(layers, row);
                }
                totalError = err;
            } while (totalError > maxError);
        } catch (NoInputLayerException | EmptyNeuralNetworkException e) {
            throw new RuntimeException(e);
        }
    }
    private float calculateError(List<Layer> layers, DataSet.DataSetRow row) {
        float err = 0;
        Layer outLayer = layers.get(layers.size() - 1);
        float[] output = outLayer.getOutput();
        error[layers.size() - 1] = new float[output.length];

        for (int current = 0; current < output.length; current++) {
            error[layers.size() - 1][current] = (row.output[current] - output[current]);
            err += Math.abs(error[layers.size() - 1][current]);

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
            Layer layer = layers.get(l);
            error[l] = new float[layer.getLayerSize()];

            for (int current = 0; current < layer.getLayerSize(); current++) {
                float e = 0;
                for (int next = 0; next < layers.get(l + 1).getLayerSize(); next++)
                    e += layers.get(l + 1).getWeights()[current + next * layer.getLayerSize()] * error[l + 1][next];

                error[l][current] = e;
                float wsum = layer.getWeightedSum()[current];
                double pow = Math.pow(1 + Math.exp(-wsum), 2);
                if (layer.getAfType() == 1) error[l][current] *= Math.exp(-wsum) / pow;
                else if (layer.getAfType() == 2) error[l][current] *= 2f * Math.exp(-wsum) / pow;
                else if (layer.getAfType() == 3) error[l][current] *= 1f / (1f + (float)Math.exp(-wsum));
                else if (layer.getAfType() == 4) error[l][current] *= wsum > 0 ? 1 : 0;
                else if (layer.getAfType() == 5) error[l][current] *= wsum > 0 ? 1 : 0.01;
                else if (layer.getAfType() == 6) error[l][current] *= ((wsum + 1) * Math.exp(-wsum) + 1) / pow;
            }
        }
        return err;
    }
    private void calculateWeights(List<Layer> layers, DataSet.DataSetRow row) {
        for (int current = 0; current < layers.get(0).getLayerSize(); current++) {
            for (int prev = 0; prev < row.input.length; prev++) {
                acceleration[0][current][prev] *= momentum;
                acceleration[0][current][prev] += (1 - momentum) * error[0][current] * row.input[prev] * getLearningSpeed();
                layers.get(0).getWeights()[current * row.input.length + prev] += acceleration[0][current][prev];
            }
            layers.get(0).getBiasWeights()[current] += error[0][current] * getLearningSpeed();
        }

        for (int l = 1; l < layers.size(); l++) {
            for (int current = 0; current < layers.get(l).getLayerSize(); current++) {
                for (int prev = 0; prev < layers.get(l - 1).getLayerSize(); prev++) {
                    acceleration[l][current][prev] *= momentum;
                    acceleration[l][current][prev] += (1 - momentum) * error[l][current] * layers.get(l - 1).getOutput()[prev] * getLearningSpeed();
                    layers.get(l).getWeights()[current * layers.get(l - 1).getLayerSize() + prev] += acceleration[l][current][prev];
                }
                layers.get(l).getBiasWeights()[current] += error[l][current] * getLearningSpeed();
            }
        }
    }

    public void setMaxError(float maxError) {
        this.maxError = maxError;
    }

    public float getMaxError() {
        return maxError;
    }

    public float getMomentum() {
        return momentum;
    }

    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }
}
