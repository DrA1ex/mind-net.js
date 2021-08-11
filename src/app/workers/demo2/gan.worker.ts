/// <reference lib="webworker" />

import * as matrix from "../../utils/matrix";
import * as color from "../../utils/color";

import {GenerativeAdversarialNetwork} from "../../neural-network/generative-adversarial";
import {COLOR_A_BIN, COLOR_B_BIN, DRAWING_DELAY, MAX_ITERATION_TIME, NetworkParams, TRAINING_BATCH_SIZE, TrainingData} from "./gan.worker.consts"


let lastDrawTime = 0;
let trainingData: TrainingData[];
let nnParams: NetworkParams = [10, [16, 32, 64], 28 * 28, [16, 16]];
let neuralNetwork = new GenerativeAdversarialNetwork(...nnParams);

addEventListener('message', ({data}) => {
    function _refresh() {
        if (data.params) {
            nnParams = data.params;
        }

        neuralNetwork = new GenerativeAdversarialNetwork(...nnParams);
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
        neuralNetwork.train(data.data, data.input ?? matrix.random(data.inputSize));

        if (i % TRAINING_BATCH_SIZE === 0 && (performance.now() - beginTime) > MAX_ITERATION_TIME) {
            break;
        }
    }

    if (performance.now() - lastDrawTime > DRAWING_DELAY) {
        draw();
    }
}

function draw() {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];

    const size = Math.floor(Math.sqrt(data.data.length))
    const dataBuffer = dataToImageBuffer(data.data);

    const genBuffer = dataToImageBuffer(neuralNetwork.generate(data.input ?? matrix.random(data.inputSize)));

    postMessage({
        type: "training_data",
        width: size,
        height: size,
        trainingData: dataBuffer,
        generatedData: genBuffer,
        gSnapshot: neuralNetwork.generator.getSnapshot(),
        dSnapshot: neuralNetwork.discriminator.getSnapshot()
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