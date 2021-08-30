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
    DRAW_SAMPLES_DIMENSION,
    DRAWING_DELAY,
    MAX_ITERATION_TIME,
    NetworkParams,
    TRAINING_BATCH_SIZE
} from "./gan.worker.consts"


let lastDrawTime = 0;
let trainingIterations = 0;
let trainingData: number[][];

let nnParams: NetworkParams = DEFAULT_NN_PARAMS;
let learningRate = DEFAULT_LEARNING_RATE;

let neuralNetwork = createNn();

function createNn() {
    const [input, genSizes, output] = nnParams;

    const generator = new NN.Models.Sequential(new NN.Optimizers.sgd(learningRate));
    generator.addLayer(new NN.Layers.Dense(input));
    for (const size of genSizes) {
        generator.addLayer(new NN.Layers.Dense(size));
    }
    generator.addLayer(new NN.Layers.Dense(output));
    generator.compile();

    const discriminator = new NN.Models.Sequential(new NN.Optimizers.sgd(learningRate));
    discriminator.addLayer(new NN.Layers.Dense(output));
    for (const size of iter.reverse(genSizes)) {
        discriminator.addLayer(new NN.Layers.Dense(size));
    }
    discriminator.addLayer(new NN.Layers.Dense(1));
    discriminator.compile();

    return new NN.Models.GAN(generator, discriminator);
}

addEventListener('message', ({data}) => {
    function _refresh() {
        if (data.params) {
            nnParams = data.params;
        }

        neuralNetwork = createNn();

        trainingIterations = 0;
        if (trainingData && trainingData.length > 0) {
            draw();
        }
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

    iter.shuffle(trainingData);

    const startTime = performance.now();
    let i = 0;
    while (++i) {
        const data = trainingData[i % trainingData.length];
        neuralNetwork.train(data, [1], nnUtils.generateInputNoise(nnParams[0]), [0]);

        if (i % TRAINING_BATCH_SIZE === 0 && (performance.now() - startTime) > MAX_ITERATION_TIME) {
            break;
        }
    }

    trainingIterations += i;

    const batchTime = performance.now() - startTime;
    console.log(`*** BATCH FINISHED with ${i} in ${batchTime.toFixed(2)}ms (${(i / batchTime * 1000).toFixed(2)} op/s)`)

    const disRealErr = nnUtils.mse(
        neuralNetwork.discriminator.compute(nnUtils.pickRandomItem(trainingData)), [1]);
    const disFakeErr = nnUtils.mse(
        neuralNetwork.discriminator.compute(
            neuralNetwork.generator.compute(nnUtils.generateInputNoise(nnParams[0]))), [0]);

    console.log(`*** DISCRIMINATOR real loss ${disRealErr.toFixed(2)}, fake loss ${disFakeErr.toFixed(2)}`);

    const t = performance.now();
    if (t - lastDrawTime > DRAWING_DELAY) {
        draw();

        lastDrawTime = t;
        console.log(`*** DRAW took ${(performance.now() - t).toFixed(2)}ms`)
    }
}

function draw() {
    const data = nnUtils.pickRandomItem(trainingData);

    const size = Math.floor(Math.sqrt(data.length))
    const dataBuffer = dataToImageBuffer(data).buffer;

    const resultDim = size * DRAW_SAMPLES_DIMENSION;
    const result = new Uint32Array(resultDim * resultDim);
    for (let i = 0; i < DRAW_SAMPLES_DIMENSION; i++) {
        for (let j = 0; j < DRAW_SAMPLES_DIMENSION; j++) {
            const inputNoise = nnUtils.generateInputNoise(nnParams[0]);
            const fake = neuralNetwork.compute(inputNoise);
            const img = dataToImageBuffer(fake);

            const startRow = i * size, startCol = j * size;
            for (let k = 0; k < img.length; k++) {
                const col = k % size;
                const row = (k - col) / size;

                result[(startRow + row) * resultDim + startCol + col] = img[k];
            }
        }
    }

    const resultBuffer = result.buffer;

    postMessage({
        type: "training_data",
        width: size,
        height: size,
        genWidth: resultDim,
        genHeight: resultDim,
        trainingData: dataBuffer,
        generatedData: resultBuffer,
        currentIteration: trainingIterations
    }, [dataBuffer, resultBuffer])
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