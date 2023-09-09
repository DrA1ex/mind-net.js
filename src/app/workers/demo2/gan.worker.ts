/// <reference lib="webworker" />

import {Color, MultiPlotChart, PlotAxisScale, PlotSeriesOverflow} from "text-graph.js";

import NN, {GanSerialization, Matrix, CommonUtils} from "../../neural-network/neural-network";
import {Matrix1D, Matrix2D} from "../../neural-network/engine/matrix";
import {LossT} from "../../neural-network/engine/loss";
import * as iter from "../../neural-network/engine/iter";
import * as color from "../../utils/color";

import {
    COLOR_A_BIN,
    COLOR_B_BIN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NN_PARAMS,
    DRAW_GRID_DIMENSION,
    DRAWING_DELAY,
    MAX_ITERATION_TIME,
    NetworkParams,
    PROGRESS_DELAY
} from "./gan.worker.consts"

let lastDrawTime = 0;
let currentIteration = 0;
let trainingData: number[][];
let batchedTrainingData: number[][][];
let batchSize = DEFAULT_BATCH_SIZE;
let batchCount = 0;

let testData: Matrix2D;
let testDataTrue: Matrix2D;
let testNoise: Matrix2D;
let testNoiseTrue: Matrix2D;
let exampleNoise: Matrix2D[];

let nnParams: NetworkParams = DEFAULT_NN_PARAMS;
let learningRate = DEFAULT_LEARNING_RATE;

let neuralNetwork = createNn();
let dashboard: MultiPlotChart;

function createNn() {
    function _createOptimizer() {
        return new NN.Optimizers.AdamOptimizer({lr: learningRate, decay: 5e-5, beta1: 0.5});
    }

    function _createGenHiddenLayer(size: number) {
        return new NN.Layers.Dense(size, {activation: "relu", weightInitializer: "xavier"});
    }

    function _createDiscriminatorHiddenLayer(size: number) {
        return new NN.Layers.Dense(size, {
            activation: new NN.Activations.LeakyReluActivation({alpha: 0.2}),
            weightInitializer: "xavier",
            options: {
                dropout: .3,
                l1WeightRegularization: 1e-5,
                l1BiasRegularization: 1e-5,
                l2WeightRegularization: 1e-4,
                l2BiasRegularization: 1e-4,
            }
        });
    }

    const [input, genSizes, output] = nnParams;
    const loss: LossT = "binaryCrossEntropy";

    const generator = new NN.Models.Sequential(_createOptimizer(), loss);
    generator.addLayer(new NN.Layers.Dense(input));
    for (const size of genSizes) {
        generator.addLayer(_createGenHiddenLayer(size));
    }
    generator.addLayer(new NN.Layers.Dense(output, {activation: "tanh"}));
    generator.compile();

    const discriminator = new NN.Models.Sequential(_createOptimizer(), loss);
    discriminator.addLayer(new NN.Layers.Dense(output));
    for (const size of iter.reverse(genSizes)) {
        discriminator.addLayer(_createDiscriminatorHiddenLayer(size));
    }
    discriminator.addLayer(new NN.Layers.Dense(1));
    discriminator.compile();

    return new NN.ComplexModels.GenerativeAdversarial(generator, discriminator, _createOptimizer(), loss);
}

function createDashboard() {
    const chart = new MultiPlotChart();

    // Loss
    chart.addPlot({
        xOffset: 0, yOffset: 0,
        width: 100, height: 21
    }, {title: "Loss", axisScale: PlotAxisScale.log});
    chart.addPlotSeries(0, {color: Color.blue});
    chart.addPlotSeries(0, {color: Color.green});

    // Learning rate
    chart.addPlot({
        xOffset: chart.plots[0].width + 1, yOffset: 0,
        width: 40, height: 10,
    }, {title: "L. rate"});
    chart.addPlotSeries(1, {color: Color.yellow});

    // Speed
    chart.addPlot({
        xOffset: chart.plots[0].width + 1, yOffset: chart.plots[1].height + 1,
        width: chart.plots[1].width, height: 10,
    }, {title: "Speed"});
    chart.addPlotSeries(2, {color: Color.red, overflow: PlotSeriesOverflow.clamp});

    return chart;
}

addEventListener('message', ({data}) => {
    function _refresh() {
        nnParams[2] = trainingData[0].length;

        testData = Array.from(iter.take(iter.shuffled(trainingData), 100));
        testDataTrue = Matrix.one_2d(testData.length, 1);
        testNoise = Matrix.random_normal_2d(100, nnParams[0], -1, 1);
        testNoiseTrue = Matrix.one_2d(testNoise.length, 1);
        exampleNoise = Matrix.fill(() => Matrix.random_normal_2d(10, nnParams[0], -1, 1), 10);

        neuralNetwork = createNn();
        dashboard = createDashboard();

        currentIteration = 0;
        batchCount = Math.ceil(trainingData.length / batchSize);

        if (trainingData && trainingData.length > 0) {
            draw();
        }

        postMessage({
            type: "progress",
            epoch: 1,
            batchNo: 1,
            batchCount,
            speed: 0,
            nnParams
        });
    }

    switch (data.type) {
        case "refresh":
            if (data.learningRate > 0) {
                learningRate = data.learningRate;
            }
            if (data?.layers?.length === 3) {
                nnParams = data.layers;
            }

            if (data?.batchSize > 0) {
                batchSize = data.batchSize;
            }

            _refresh();
            break;

        case "set_data":
            trainingData = data.data;
            for (let i = 0; i < trainingData.length; i++) {
                for (let j = 0; j < trainingData[i].length; j++) {
                    trainingData[i][j] = (0.5 - trainingData[i][j]) * 2;
                }
            }

            _refresh();
            break;

        case "dump":
            const modelData = GanSerialization.save(neuralNetwork);
            postMessage({type: "model_dump", dump: modelData});
            break;

        case "load_dump":
            neuralNetwork = GanSerialization.load(data.dump);
            dashboard = createDashboard();

            currentIteration = 0;
            if (trainingData && trainingData.length > 0) {
                draw();
            }

            break;
    }
});

function trainBatch() {
    if (!neuralNetwork || !trainingData || trainingData.length === 0) {
        return
    }

    const startTime = performance.now();
    let progressLastTime = startTime;
    let batches = 0
    let batchesCntToCheck = 0;

    while (true) {
        if (currentIteration % batchCount === 0) {
            batchedTrainingData = Array.from(iter.partition(iter.shuffled(trainingData), batchSize));
            neuralNetwork.beforeTrain();
        }

        const input = batchedTrainingData[currentIteration % batchCount];
        neuralNetwork.trainBatch(input);

        if (++batches > batchesCntToCheck) {
            neuralNetwork.afterTrain();

            const elapsed = performance.now() - progressLastTime;
            const batchTime = elapsed / batches;
            batchesCntToCheck = PROGRESS_DELAY / batchTime;
            const speed = 1000 / batchTime;

            postMessage({
                type: "progress",
                epoch: Math.floor(currentIteration / batchCount) + 1,
                batchNo: currentIteration % batchCount + 1,
                batchCount,
                speed
            });

            const {loss: noiseLoss} = neuralNetwork.ganChain.evaluate(testNoise, testNoiseTrue);
            const {loss: realLoss} = neuralNetwork.discriminator.evaluate(testData, testDataTrue);

            const loss = (noiseLoss + realLoss) / 2;
            dashboard.plots[0].title = `Loss: ${loss.toFixed(6)}`;
            dashboard.addSeriesEntry(0, 0, noiseLoss);
            dashboard.addSeriesEntry(0, 1, realLoss);

            dashboard.plots[1].title = `L. rate: ${neuralNetwork.generator.optimizer.lr.toFixed(6)}`;
            dashboard.addSeriesEntry(1, 0, neuralNetwork.generator.optimizer.lr);
            dashboard.addSeriesEntry(2, 0, speed);

            batches = 0;
            progressLastTime = performance.now();
        }

        ++currentIteration;

        if ((performance.now() - startTime) > MAX_ITERATION_TIME) {
            break;
        }
    }

    postMessage({
        type: "progress",
        epoch: Math.floor(currentIteration / batchCount) + 1,
        batchNo: currentIteration % batchCount + 1,
    });

    const t = performance.now();
    if (t - lastDrawTime > DRAWING_DELAY) {
        draw();

        lastDrawTime = t;

        const chart = dashboard.paint();
        console.clear();
        console.log(chart)
    }
}


function draw() {
    const size = Math.floor(Math.sqrt(nnParams[2]))
    const gridSize = size * DRAW_GRID_DIMENSION;

    const dataSamples = drawGridSample(DRAW_GRID_DIMENSION, size,
        () => CommonUtils.pickRandomItem(trainingData));

    const genSamples = drawGridSample(DRAW_GRID_DIMENSION, size, (i, j) => {
        return neuralNetwork.compute(exampleNoise[i][j]);
    })

    postMessage({
        type: "training_data",
        width: gridSize,
        height: gridSize,
        trainingData: dataSamples,
        generatedData: genSamples,
        currentIteration: currentIteration
    }, [dataSamples, genSamples])
}

function drawGridSample(gridDimension: number, sampleDimension: number, fn: (i: number, j: number) => Matrix1D): ArrayBuffer {
    const resultDim = sampleDimension * gridDimension;
    const result = new Uint32Array(resultDim * resultDim);
    for (let i = 0; i < gridDimension; i++) {
        for (let j = 0; j < gridDimension; j++) {
            const img = dataToImageBuffer(fn(i, j));

            const startRow = i * sampleDimension, startCol = j * sampleDimension;
            for (let k = 0; k < img.length; k++) {
                const col = k % sampleDimension;
                const row = (k - col) / sampleDimension;

                result[(startRow + row) * resultDim + startCol + col] = img[k];
            }
        }
    }

    return result.buffer;
}

function dataToImageBuffer(data: Matrix1D): Uint32Array {
    const state = new Uint32Array(data.length)
    for (let i = 0; i < data.length; i++) {
        state[i] = color.getLinearColorBin(COLOR_A_BIN, COLOR_B_BIN, (data[i] / 2 + 0.5));
    }

    return state;
}

function runTrainingPass() {
    try {
        trainBatch()
    } finally {
        if (!neuralNetwork || !trainingData || trainingData.length === 0) {
            setTimeout(runTrainingPass, 1000);
        } else {
            setTimeout(runTrainingPass, 0);
        }
    }
}

setTimeout(runTrainingPass, 0);