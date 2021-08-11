/// <reference lib="webworker" />

import {SequentialNetwork} from "../../neural-network/sequential";
import * as nnUtils from "../../neural-network/utils";
import * as color from "../../utils/color";
import {Matrix1D} from "../../utils/matrix";
import {
    COLOR_A_BIN,
    COLOR_B_BIN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NN_LAYERS,
    DESIRED_RESOLUTION_X,
    DESIRED_RESOLUTION_Y,
    DRAWING_DELAY,
    MAX_ITERATION_TIME,
    MAX_TRAINING_ITERATION,
    Point,
    RESOLUTION_SCALE,
    TRAINING_BATCH_SIZE,
} from "./nn.worker.consts"

let neuralNetwork = new SequentialNetwork(2, ...DEFAULT_NN_LAYERS, 1);
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

            neuralNetwork = new SequentialNetwork(2, ...newLayersConfig, 1);
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

        if (iterationCnt % TRAINING_BATCH_SIZE == 0 && performance.now() - startTime > MAX_ITERATION_TIME) {
            break;
        }
    }

    currentTrainIterations += iterationCnt;
    if (currentTrainIterations >= MAX_TRAINING_ITERATION) {
        lastDraw = 0;
        console.log('*** DATA SET TRAINING FINISHED ***');
        nnUtils.print(neuralNetwork, trainingData)

        sendCurrentState(2);
    } else {
        sendCurrentState();
    }
}

function sendCurrentState(scale: number = RESOLUTION_SCALE) {
    const t = performance.now();
    if (t - lastDraw < DRAWING_DELAY) {
        return;
    }

    const xStep = 1 / DESIRED_RESOLUTION_X / scale;
    const yStep = 1 / DESIRED_RESOLUTION_Y / scale;

    const xSteps = Math.ceil(1 / xStep),
        ySteps = Math.ceil(1 / yStep);

    const state = new Uint32Array(xSteps * ySteps);
    for (let x = 0; x < xSteps; x++) {
        for (let y = 0; y < ySteps; y++) {
            const result = neuralNetwork.compute([x * xStep, y * yStep]);
            state[y * xSteps + x] = color.getLinearColorBin(COLOR_A_BIN, COLOR_B_BIN, result[0]);
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