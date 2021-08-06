/// <reference lib="webworker" />

import {NeuralNetwork} from "../neural-network/neural_network";
import * as nnUtils from "../neural-network/utils";
import {Matrix1D} from "../utils/matrix";
import {
    COLOR_PATTERN_BIN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NN_LAYERS,
    DRAWING_DELAY,
    MAX_ITERATION_TIME,
    MAX_TRAINING_ITERATION,
    Point,
    X_STEP,
    Y_STEP
} from "./nn.worker.consts"

let neuralNetwork = new NeuralNetwork(2, ...DEFAULT_NN_LAYERS, 1);
neuralNetwork.learningRate = DEFAULT_LEARNING_RATE;

let points: Point[] = [];

let currentTrainIterations = 0;
let lastDraw = 0;

addEventListener('message', ({data}) => {
    switch (data.type) {
        case "add_point":
            points.push(data.point as Point);
            currentTrainIterations = 0;
            break;

        case "set_points":
            points = data.points as Point[];
            currentTrainIterations = 0;
            break;

        case "refresh":
            const newLayersConfig = data.config?.layers || DEFAULT_NN_LAYERS;
            const newLearningRateConfig = data.config?.learningRate || DEFAULT_LEARNING_RATE;

            neuralNetwork = new NeuralNetwork(2, ...newLayersConfig, 1);
            neuralNetwork.learningRate = newLearningRateConfig;
            currentTrainIterations = 0;
    }
});

function trainBatch() {
    const iterationsLeft = MAX_TRAINING_ITERATION - currentTrainIterations;
    if (iterationsLeft <= 0 || points.length === 0) {
        return;
    }

    const startTime = performance.now();
    const trainingData: [Matrix1D, Matrix1D][] = points.map(p => ([[p.x, p.y], [p.type]]));

    let iterationCnt;
    for (iterationCnt = 0; iterationCnt < iterationsLeft; iterationCnt++) {
        const data = trainingData[Math.floor(Math.random() * trainingData.length)];
        neuralNetwork.train(data[0], data[1]);

        if (iterationCnt % 10000 == 0 && performance.now() - startTime > MAX_ITERATION_TIME) {
            break;
        }
    }

    currentTrainIterations += iterationCnt;
    if (currentTrainIterations >= MAX_TRAINING_ITERATION) {
        lastDraw = 0;
        console.log('*** DATA SET TRAINING FINISHED ***');
        nnUtils.print(neuralNetwork, trainingData)
    }

    sendCurrentState()
}

function sendCurrentState() {
    const t = performance.now();
    if (t - lastDraw < DRAWING_DELAY) {
        return;
    }

    const xSteps = Math.ceil(1 / X_STEP),
        ySteps = Math.ceil(1 / Y_STEP);

    const state = new Uint32Array(xSteps * ySteps);
    for (let x = 0; x < xSteps; x++) {
        for (let y = 0; y < ySteps; y++) {
            const result = neuralNetwork.compute([x * X_STEP, y * Y_STEP]);
            state[y * xSteps + x] = COLOR_PATTERN_BIN | ((result[0] * 0xff & 0xff) << 8);
        }
    }


    const message = {
        type: "training_data",
        iteration: currentTrainIterations,
        state: state.buffer,
        nnSnapshot: neuralNetwork.getSnapshot(),
        width: xSteps,
        height: ySteps,
        t: performance.now()
    };

    lastDraw = t;
    postMessage(message, [state.buffer]);

    console.log(`*** Computing ${(performance.now() - t).toFixed(2)}ms`);
}

function runTrainingPass() {
    try {
        trainBatch()
    } finally {
        if (currentTrainIterations < MAX_TRAINING_ITERATION && points.length > 0) {
            setTimeout(runTrainingPass, 3);
        } else {
            setTimeout(runTrainingPass, 1000);
        }
    }
}

setTimeout(runTrainingPass, 0);