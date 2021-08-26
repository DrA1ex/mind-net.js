/// <reference lib="webworker" />

import * as matrix from "../../neural-network/engine/matrix";
import * as color from "../../utils/color";
import * as nnUtils from "../../neural-network/utils";

import {GenerativeAdversarialNetwork} from "../../neural-network/generative-adversarial";
import {
    COLOR_A_BIN,
    COLOR_B_BIN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NN_PARAMS,
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

let activation = nnUtils.sigmoid;
let activationDer = nnUtils.der_sigmoid;

// TODO:
//let activation = (v: number) => nnUtils.leakyReLU(v, 0.2);
//let activationDer = (v: matrix.Matrix1D) => nnUtils.der_leakyReLU(v, 0.2);

let neuralNetwork = createNn();

function createNn() {
    const nn = new GenerativeAdversarialNetwork(...nnParams);
    nn.learningRate = learningRate;

    nn.generator.activationFn = activation;
    nn.generator.activationDerivativeFn = activationDer;

    nn.discriminator.activationFn = activation;
    nn.discriminator.activationDerivativeFn = activationDer;

    return nn;
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
            if (data?.layers?.length === 4) {
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

    const beginTime = performance.now();
    let i = 0;
    while (++i) {
        const data = trainingData[Math.floor(Math.random() * trainingData.length)];
        neuralNetwork.train(data, matrix.random(neuralNetwork.generator.layers[0].neuronCnt));

        if (i % TRAINING_BATCH_SIZE === 0 && (performance.now() - beginTime) > MAX_ITERATION_TIME) {
            break;
        }
    }

    trainingIterations += i;

    console.log(`*** BATCH TRAINING finished with ${i} iterations, took ${(performance.now() - beginTime).toFixed(2)}ms`)

    const t = performance.now();
    if (t - lastDrawTime > DRAWING_DELAY) {
        draw();

        lastDrawTime = t;
        console.log(`*** DRAW took ${(performance.now() - t).toFixed(2)}ms`)
    }
}

function draw() {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];

    const size = Math.floor(Math.sqrt(data.length))
    const dataBuffer = dataToImageBuffer(data);

    const genBuffer = dataToImageBuffer(neuralNetwork.generate(matrix.random(neuralNetwork.generator.layers[0].neuronCnt)));

    postMessage({
        type: "training_data",
        width: size,
        height: size,
        trainingData: dataBuffer,
        generatedData: genBuffer,
        currentIteration: trainingIterations
    }, [dataBuffer, genBuffer])
}

function dataToImageBuffer(data: number[]): ArrayBuffer {
    const state = new Uint32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        state[i] = color.getLinearColorBin(COLOR_A_BIN, COLOR_B_BIN, data[i]);
    }

    return state.buffer;
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