package ru.alexander1248.nnlib.core.learning.teacher;

import com.aparapi.exception.CompileFailedException;
import com.aparapi.internal.kernel.KernelManager;
import ru.alexander1248.nnlib.core.layers.Layer;
import ru.alexander1248.nnlib.core.exceptions.EmptyNeuralNetworkException;
import ru.alexander1248.nnlib.core.exceptions.NoInputLayerException;
import ru.alexander1248.nnlib.core.kernels.ThreadKernel;
import ru.alexander1248.nnlib.core.kernels.learning.teacher.RandomizingKernel;
import ru.alexander1248.nnlib.core.learning.DataSet;
import ru.alexander1248.nnlib.core.types.ThreadingType;

import java.util.ArrayList;
import java.util.List;

import static ru.alexander1248.nnlib.core.types.ThreadingType.CPU;

public class AnnealingLearning extends TeacherLearning {
    protected ThreadKernel cpuWeights;
    protected RandomizingKernel gpuWeights;

    protected int l = 0;
    protected float[] layerInput;
    public AnnealingLearning() {
        setThreadingType(CPU);
    }

    @Override
    public void learn() {
        try {
            iteration = 0;
            float initT = 10;
            float temp = 1;
            do {
                for (DataSet.DataSetRow row : dataSet.getRows()) {
                    List<Layer> layers = network.getLayers();
                    List<Layer> old = new ArrayList<>();
                    for (Layer layer : network.getLayers()) old.add(layer.clone());


                    network.setInput(row.input);
                    network.calculate();

                    float preErr = calculateError(layers, row);
                    calculateWeights(layers, row);

                    network.setInput(row.input);
                    network.calculate();
                    float postErr = calculateError(layers, row);

                    if (preErr < postErr) {
                        network.getLayers().clear();
                        for (Layer layer : old) network.getLayers().add(layer);
                    }
                    else {
                        double p = Math.exp((preErr - postErr) / temp);
                        if (Math.random() > p) {
                            network.getLayers().clear();
                            for (Layer layer : old) network.getLayers().add(layer);
                        }
                    }
                }

                float err = 0;
                List<Layer> layers = network.getLayers();
                for (DataSet.DataSetRow row : dataSet.getRows()) {
                    network.setInput(row.input);
                    network.calculate();
                    err += calculateError(layers, row);
                }
                totalError = err;
                iteration++;
                temp = (float) (initT / Math.sqrt(iteration));
            } while (totalError > maxError && (maxIterations < 0 || iteration < maxIterations));
        } catch (NoInputLayerException | EmptyNeuralNetworkException e) {
            throw new RuntimeException(e);
        }
    }

    private float calculateError(List<Layer> layers, DataSet.DataSetRow row) {
        float err = 0;
        float[] output = layers.get(layers.size() - 1).getOutput();
        for (int current = 0; current < output.length; current++)
            err += Math.abs(row.output[current] - output[current]);
        return err;
    }
    private void calculateWeights(List<Layer> layers, DataSet.DataSetRow row) {
        switch (workingType) {
            case CPU -> {
                l = 0;
                layerInput = row.input;
                cpuWeights.execute(layers.get(0).getLayerSize());
                for (l = 1; l < layers.size(); l++) {
                    layerInput = layers.get(l - 1).getOutput();
                    cpuWeights.execute(layers.get(l).getLayerSize());
                }
            }
            case GPU -> {
                gpuWeights.weights = network.getLayers().get(0).getWeights();
                gpuWeights.biasWeights = network.getLayers().get(0).getBiasWeights();

                gpuWeights.layerSize = network.getLayers().get(0).getLayerSize();
                gpuWeights.prevLayerSize = row.input.length;
                gpuWeights.learningSpeed = getLearningSpeed();
                gpuWeights.random = Math.random() * 100;

                gpuWeights.execute(gpuWeights.layerSize);

                network.getLayers().get(0).setWeights(gpuWeights.weights);
                network.getLayers().get(0).setBiasWeights(gpuWeights.biasWeights);
                for (int l = 1; l < layers.size(); l++) {
                    gpuWeights.weights = network.getLayers().get(l).getWeights();
                    gpuWeights.biasWeights = network.getLayers().get(l).getBiasWeights();

                    gpuWeights.layerSize = network.getLayers().get(l).getLayerSize();
                    gpuWeights.prevLayerSize = network.getLayers().get(l - 1).getLayerSize();
                    gpuWeights.learningSpeed = getLearningSpeed();
                    gpuWeights.random = Math.random() * 100;

                    gpuWeights.execute(gpuWeights.layerSize);

                    network.getLayers().get(l).setWeights(gpuWeights.weights);
                    network.getLayers().get(l).setBiasWeights(gpuWeights.biasWeights);
                }
            }
        }

    }


    public void setThreadingType(ThreadingType threadingType) {
        this.workingType = threadingType;
        switch (workingType) {
            case CPU -> cpuWeights = new ThreadKernel(Runtime.getRuntime().availableProcessors() / 2) {
                @Override
                public void run(int gid) {
                    for (int prev = 0; prev < layerInput.length; prev++)
                        network.getLayers().get(l).getWeights()[gid * layerInput.length + prev] += (Math.random() * 2 - 1) * getLearningSpeed();

                    network.getLayers().get(l).getBiasWeights()[gid] += (Math.random() * 2 - 1) * getLearningSpeed();
                }
            };
            case GPU -> {
                gpuWeights = new RandomizingKernel();
                try {
                    gpuWeights.compile(KernelManager.instance().getDefaultPreferences().getPreferredDevices(null).get(0));
                } catch (CompileFailedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
