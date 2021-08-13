/// <reference lib="webworker" />

import * as matrix from "../../utils/matrix";
import * as color from "../../utils/color";

import {GenerativeAdversarialNetwork} from "../../neural-network/generative-adversarial";
import {COLOR_A_BIN, COLOR_B_BIN, DRAWING_DELAY, MAX_ITERATION_TIME, NetworkParams, TRAINING_BATCH_SIZE} from "./gan.worker.consts"


let lastDrawTime = 0;
let trainingIterations = 0;
let trainingData: matrix.Matrix1D[];
let nnParams: NetworkParams = [1, [16, 16], 28 * 28, [16, 16]];
let neuralNetwork = new GenerativeAdversarialNetwork(...nnParams);

addEventListener('message', ({data}) => {
    function _refresh() {
        if (data.params) {
            nnParams = data.params;
        }

        neuralNetwork = new GenerativeAdversarialNetwork(...nnParams);
        trainingIterations = 0;
        if (trainingData && trainingData.length > 0) {
            draw();
        }
    }

    switch (data.type) {
        case "refresh":
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
        neuralNetwork.train(data, matrix.random(1));

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

    const genBuffer = dataToImageBuffer(neuralNetwork.generate(matrix.random(1)));

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