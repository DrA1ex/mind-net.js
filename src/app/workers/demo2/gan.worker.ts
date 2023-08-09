/// <reference lib="webworker" />

import * as iter from "../../neural-network/engine/iter";
import * as color from "../../utils/color";
import * as nnUtils from "../../neural-network/utils";
import NN, {Matrix} from "../../neural-network/neural-network";
import {LossT} from "../../neural-network/engine/loss";

import {
    COLOR_A_BIN,
    COLOR_B_BIN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NN_PARAMS,
    DRAW_GRID_DIMENSION,
    DRAWING_DELAY,
    MAX_ITERATION_TIME,
    NetworkParams,
    PROGRESS_DELAY,
    TRAINING_BATCH_SIZE,
    DEFAULT_BATCH_SIZE
} from "./gan.worker.consts"

let lastDrawTime = 0;
let currentIteration = 0;
let trainingData: number[][];
let batchedTrainingData: number[][][];
let batchSize = DEFAULT_BATCH_SIZE;
let batchCount = 0;

let nnParams: NetworkParams = DEFAULT_NN_PARAMS;
let learningRate = DEFAULT_LEARNING_RATE;

let neuralNetwork = createNn();

function createNn() {
    function _createOptimizer() {
        return new NN.Optimizers.adam(learningRate, 0, 0.5);
    }

    function _createGenHiddenLayer(size: number) {
        return new NN.Layers.Dense(size, "relu", "xavier");
    }

    function _createDiscriminatorHiddenLayer(size: number) {
        return new NN.Layers.Dense(size, new NN.Activations.leakyRelu(0.2),
            "xavier", "zero", {dropout: .3});
    }

    const [input, genSizes, output] = nnParams;
    const loss: LossT = "binaryCrossEntropy";

    const generator = new NN.Models.Sequential(_createOptimizer(), loss);
    generator.addLayer(new NN.Layers.Dense(input));
    for (const size of genSizes) {
        generator.addLayer(_createGenHiddenLayer(size));
    }
    generator.addLayer(new NN.Layers.Dense(output, "tanh"));
    generator.compile();

    const discriminator = new NN.Models.Sequential(_createOptimizer(), loss);
    discriminator.addLayer(new NN.Layers.Dense(output));
    for (const size of iter.reverse(genSizes)) {
        discriminator.addLayer(_createDiscriminatorHiddenLayer(size));
    }
    discriminator.addLayer(new NN.Layers.Dense(1));
    discriminator.compile();

    return new NN.Models.GAN(generator, discriminator, _createOptimizer(), loss);
}


addEventListener('message', ({data}) => {
    function _refresh() {
        if (data.params) {
            nnParams = data.params;
        }

        nnParams[2] = trainingData[0].length;

        neuralNetwork = createNn();

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

            postMessage({
                type: "progress",
                epoch: Math.floor(currentIteration / batchCount) + 1,
                batchNo: currentIteration % batchCount + 1,
                batchCount,
                speed: batchCount * 1000 / batchTime
            });

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
        console.log(`*** DRAW took ${(performance.now() - t).toFixed(2)}ms`)
    }
}


function draw() {
    const size = Math.floor(Math.sqrt(nnParams[2]))
    const gridSize = size * DRAW_GRID_DIMENSION;

    const dataSamples = drawGridSample(DRAW_GRID_DIMENSION, size,
        () => nnUtils.pickRandomItem(trainingData));

    const genSamples = drawGridSample(DRAW_GRID_DIMENSION, size, () => {
        return neuralNetwork.compute(Matrix.random_normal_1d(nnParams[0]));
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

function drawGridSample(gridDimension: number, sampleDimension: number, fn: (i: number, j: number) => number[]): ArrayBuffer {
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

function dataToImageBuffer(data: number[]): Uint32Array {
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