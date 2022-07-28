package ru.alexander.nnlib.learning;

import com.aparapi.exception.CompileFailedException;
import com.aparapi.internal.kernel.KernelManager;
import ru.alexander.nnlib.Layer;
import ru.alexander.nnlib.exceptions.EmptyNeuralNetworkException;
import ru.alexander.nnlib.exceptions.NoInputLayerException;
import ru.alexander.nnlib.DataSet;
import ru.alexander.nnlib.kernels.AcceleratedWeightsKernel;
import ru.alexander.nnlib.kernels.ErrorKernel;
import ru.alexander.nnlib.kernels.ThreadKernel;
import ru.alexander.nnlib.kernels.WeightsKernel;
import ru.alexander.nnlib.types.ThreadingType;

import java.util.List;

public class MomentumBackPropagation extends BackPropagation {

    private float momentum = 0;
    private float[][] acceleration;
    public MomentumBackPropagation() {
        setThreadingType(ThreadingType.CPU);
    }

    @Override
    public void learn(DataSet dataSet) {
        try {
            List<Layer> layers = network.getLayers();
            error = new float[layers.size()][];
            iteration = 0;
            acceleration = new float[layers.size()][];
            for (int i =0; i < layers.size(); i++) {
                acceleration[i] = new float[layers.get(i).getWeights().length];
                for (int j = 0; j < layers.get(i).getWeights().length; j++) {
                    acceleration[i][j] = 0;
                }
            }

            do {
                float err = 0;
                for (DataSet.DataSetRow row : dataSet.getRows()) {
                    network.setInput(row.input);
                    network.calculate();

                    err += calculateError(layers, row);
                    calculateWeights(layers, row);
                }
                totalError = err;
                iteration++;
            } while (totalError > maxError && (maxIterations < 0 || iteration < maxIterations));
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

        switch (workingType) {
            case CPU: {
                for (l = layers.size() - 2; l >= 0; l--) {
                    int layerSize = network.getLayers().get(l).getLayerSize();
                    error[l] = new float[layerSize];
                    cpuError.execute(layerSize);
                }
            }
            case GPU: {
                for (l = layers.size() - 2; l >= 0; l--) {
                    int layerSize = network.getLayers().get(l).getLayerSize();

                    gpuError.weights = network.getLayers().get(l).getWeights();
                    gpuError.weightedSum = network.getLayers().get(l).getWeightedSum();
                    gpuError.error = new float[layerSize];
                    gpuError.nextError = error[l + 1];

                    gpuError.layerSize = network.getLayers().get(l).getLayerSize();
                    gpuError.nextLayerSize = network.getLayers().get(l + 1).getLayerSize();
                    gpuError.afType = network.getLayers().get(l).getAfType();

                    gpuError.execute(layerSize);

                    error[l] = gpuError.error;
                }
            }
        }
        return err;
    }
    private void calculateWeights(List<Layer> layers, DataSet.DataSetRow row) {
        switch (workingType) {
            case CPU: {
                l = 0;
                layerInput = row.input;
                cpuWeights.execute(layers.get(0).getLayerSize());

                for (l = 1; l < layers.size(); l++) {
                    layerInput = layers.get(l - 1).getOutput();
                    cpuWeights.execute(layers.get(l).getLayerSize());
                }
            }
            case GPU: {
                gpuWeights.weights = network.getLayers().get(0).getWeights();
                gpuWeights.biasWeights = network.getLayers().get(0).getBiasWeights();
                gpuWeights.input = row.input;
                gpuWeights.error = error[0];
                ((AcceleratedWeightsKernel)gpuWeights).acceleration = acceleration[0];
                ((AcceleratedWeightsKernel)gpuWeights).momentum = momentum;

                gpuWeights.layerSize = network.getLayers().get(0).getLayerSize();
                gpuWeights.prevLayerSize = row.input.length;
                gpuWeights.learningSpeed = getLearningSpeed();

                gpuWeights.execute(gpuWeights.layerSize);

                network.getLayers().get(0).setWeights(gpuWeights.weights);
                network.getLayers().get(0).setBiasWeights(gpuWeights.biasWeights);

                for (int l = 1; l < layers.size(); l++) {
                    gpuWeights.weights = network.getLayers().get(l).getWeights();
                    gpuWeights.biasWeights = network.getLayers().get(l).getBiasWeights();
                    gpuWeights.input = network.getLayers().get(l - 1).getOutput();
                    gpuWeights.error = error[l];
                    ((AcceleratedWeightsKernel)gpuWeights).acceleration = acceleration[l];
                    ((AcceleratedWeightsKernel)gpuWeights).momentum = momentum;

                    gpuWeights.layerSize = network.getLayers().get(l).getLayerSize();
                    gpuWeights.prevLayerSize = network.getLayers().get(l - 1).getLayerSize();
                    gpuWeights.learningSpeed = getLearningSpeed();

                    gpuWeights.execute(gpuWeights.layerSize);

                    network.getLayers().get(l).setWeights(gpuWeights.weights);
                    network.getLayers().get(l).setBiasWeights(gpuWeights.biasWeights);
                }
            }
        }

    }

    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }

    public void setThreadingType(ThreadingType threadingType) {
        this.workingType = threadingType;
        switch (workingType) {
            case CPU: {
                cpuError = new ThreadKernel(Runtime.getRuntime().availableProcessors() / 2) {
                    @Override
                    public void run(int gid) {
                        Layer layer = network.getLayers().get(l);
                        float e = 0;
                        for (int next = 0; next < network.getLayers().get(l + 1).getLayerSize(); next++)
                            e += network.getLayers().get(l + 1).getWeights()[gid + next * layer.getLayerSize()] * error[l + 1][next];

                        error[l][gid] = e;
                        float wsum = layer.getWeightedSum()[gid];
                        double pow = Math.pow(1 + Math.exp(-wsum), 2);
                        if (layer.getAfType() == 1) error[l][gid] *= Math.exp(-wsum) / pow;
                        else if (layer.getAfType() == 2) error[l][gid] *= 2f * Math.exp(-wsum) / pow;
                        else if (layer.getAfType() == 3) error[l][gid] *= 1f / (1f + (float)Math.exp(-wsum));
                        else if (layer.getAfType() == 4) error[l][gid] *= wsum > 0 ? 1 : 0;
                        else if (layer.getAfType() == 5) error[l][gid] *= wsum > 0 ? 1 : 0.01;
                        else if (layer.getAfType() == 6) error[l][gid] *= ((wsum + 1) * Math.exp(-wsum) + 1) / pow;
                    }
                };

                cpuWeights = new ThreadKernel(Runtime.getRuntime().availableProcessors() / 2) {
                    @Override
                    public void run(int gid) {
                        for (int prev = 0; prev < layerInput.length; prev++) {
                            acceleration[l][gid * layerInput.length + prev] *= momentum;
                            acceleration[l][gid * layerInput.length + prev] += (1 - momentum) * error[l][gid] * layerInput[prev] * getLearningSpeed();
                            network.getLayers().get(l).getWeights()[gid * layerInput.length + prev] += acceleration[l][gid * layerInput.length + prev];
                        }
                        network.getLayers().get(l).getBiasWeights()[gid] += error[l][gid] * getLearningSpeed();
                    }
                };
            }
            case GPU: {
                gpuError = new ErrorKernel();
                gpuWeights = new AcceleratedWeightsKernel();

                try {
                    gpuError.compile(KernelManager.instance().getDefaultPreferences().getPreferredDevices(null).get(0));
                    gpuWeights.compile(KernelManager.instance().getDefaultPreferences().getPreferredDevices(null).get(0));
                } catch (CompileFailedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
