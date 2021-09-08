/// <reference lib="webworker" />

import * as iter from "../../neural-network/engine/iter";
import * as color from "../../utils/color";
import * as nnUtils from "../../neural-network/utils";
import NN from "../../neural-network/neural-network";

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
    TRAINING_BATCH_SIZE
} from "./gan.worker.consts"

let lastDrawTime = 0;
let currentIteration = 0;
let trainingData: number[][];
let batchedTrainingData: number[][][];
let batchSize = 0;
let batchCount = 0;

let nnParams: NetworkParams = DEFAULT_NN_PARAMS;
let learningRate = DEFAULT_LEARNING_RATE;

let neuralNetwork = createNn();

function createNn() {
    function _createOptimizer() {
        return new NN.Optimizers.sgd(learningRate);
    }

    function _createHiddenLayer(size: number) {
        return new NN.Layers.Dense(size, new NN.Activations.leakyRelu(0.2))
    }

    const [input, genSizes, output] = nnParams;

    const generator = new NN.Models.Sequential(_createOptimizer());
    generator.addLayer(new NN.Layers.Dense(input));
    for (const size of genSizes) {
        generator.addLayer(_createHiddenLayer(size));
    }
    generator.addLayer(new NN.Layers.Dense(output));
    generator.compile();

    const discriminator = new NN.Models.Sequential(_createOptimizer());
    discriminator.addLayer(new NN.Layers.Dense(output));
    for (const size of iter.reverse(genSizes)) {
        discriminator.addLayer(_createHiddenLayer(size));
    }
    discriminator.addLayer(new NN.Layers.Dense(1));
    discriminator.compile();

    return new NN.Models.GAN(generator, discriminator, _createOptimizer());
}

addEventListener('message', ({data}) => {
    function _refresh() {
        if (data.params) {
            nnParams = data.params;
        }

        neuralNetwork = createNn();

        currentIteration = 0;
        batchSize = Math.max(1, Math.floor(trainingData.length / 200));
        batchCount = Math.ceil(trainingData.length / batchSize);

        if (trainingData && trainingData.length > 0) {
            draw();
        }

        postMessage({
            type: "progress",
            epoch: 1,
            batchNo: 1,
            batchCount,
            speed: 0
        });
    }

    switch (data.type) {
        case "refresh":
            if (data.learningRate) {
                learningRate = data.learningRate;
            }
            if (data?.layers?.length === 3) {
                nnParams = data.layers;
            }

            _refresh();
            break;

        case "set_data":
            trainingData = data.data;

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
        }

        neuralNetwork.trainBatch(batchedTrainingData[currentIteration % batchCount]);

        if (++batches > batchesCntToCheck) {
            const elapsed = performance.now() - progressLastTime;
            const batchTime = elapsed / batches;
            batchesCntToCheck = PROGRESS_DELAY / batchTime;

            postMessage({
                type: "progress",
                epoch: Math.floor(currentIteration / batchCount) + 1,
                batchNo: currentIteration % batchCount + 1,
                batchCount,
                speed: 1000 / batchTime
            });

            batches = 0;
            progressLastTime = performance.now();
        }

        if (++currentIteration % TRAINING_BATCH_SIZE === 0 && (performance.now() - startTime) > MAX_ITERATION_TIME) {
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
        const inputNoise = nnUtils.generateInputNoise(nnParams[0]);
        return neuralNetwork.compute(inputNoise);
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
        state[i] = color.getLinearColorBin(COLOR_A_BIN, COLOR_B_BIN, data[i]);
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