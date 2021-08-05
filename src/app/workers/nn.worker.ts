/// <reference lib="webworker" />

import {NeuralNetwork} from "../nn/neural_network";
import * as nnUtils from "../nn/utils";
import {ITERATIONS_PER_CALL, MAX_TRAINING_ITERATION, Point, X_STEP, Y_STEP} from "./nn.worker.consts"
import {Matrix1D} from "../utils/matrix";

let neuralNetwork = new NeuralNetwork(2, 5, 5, 1);
let points: Point[] = [];

let currentTrainIterations = 0;

addEventListener('message', ({data}) => {
    switch (data.type) {
        case "add_point":
            points.push(data.point as Point)
            currentTrainIterations = 0;
            break;

        case "set_points":
            points = data.points as Point[];
            currentTrainIterations = 0;
            break;

        case "refresh":
            const newLayersConfig = data.config?.layers || [5, 5];
            const newLearningRateConfig = data.config?.learningRate || 0.01;

            neuralNetwork = new NeuralNetwork(2, ...newLayersConfig, 1);
            neuralNetwork.learningRate = newLearningRateConfig;
            currentTrainIterations = 0;
    }
});

async function sendCurrentState() {
    const xSteps = Math.ceil(1 / X_STEP),
        ySteps = Math.ceil(1 / Y_STEP);

    const state = new Array(xSteps * ySteps);

    for (let x = 0; x < xSteps; x++) {
        for (let y = 0; y < ySteps; y++) {
            const result = neuralNetwork.compute([x * X_STEP, y * Y_STEP]);
            state[x * ySteps + y] = [...result];
        }
    }

    postMessage({
        type: "training_data",
        iteration: currentTrainIterations,
        state: state
    });
}

async function train() {
    if (currentTrainIterations >= MAX_TRAINING_ITERATION || points.length === 0) {
        setTimeout(train, 300);
        return;
    }

    const startTime = performance.now();
    const trainingData: [Matrix1D, Matrix1D][] = points.map(p => ([[p.x, p.y], [p.type]]));

    for (let i = 0; i < ITERATIONS_PER_CALL; i++) {
        const data = trainingData[Math.floor(Math.random() * trainingData.length)];
        neuralNetwork.train(data[0], data[1]);
    }

    nnUtils.print(neuralNetwork, trainingData.slice(0, 1))
    currentTrainIterations += ITERATIONS_PER_CALL;

    console.log(`*** TRAINING OVER ${points.length} SET WITH ${ITERATIONS_PER_CALL} ITERATIONS FINISHED IN ${(performance.now() - startTime).toFixed(2)}ms`);

    await sendCurrentState()

    if (currentTrainIterations >= MAX_TRAINING_ITERATION) {
        console.log('*** DATA SET TRAINING FINISHED ***');
        nnUtils.print(neuralNetwork, trainingData)
    }

    setTimeout(train, 1);
}

setTimeout(train, 1);
