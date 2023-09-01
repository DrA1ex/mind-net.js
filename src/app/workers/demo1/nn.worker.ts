/// <reference lib="webworker" />

import * as matrix from "../../neural-network/engine/matrix";
import * as color from "../../utils/color";
import * as logUtils from "../../utils/log";
import {
    COLOR_A_BIN,
    COLOR_B_BIN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NN_LAYERS,
    DESIRED_LOSS,
    DESIRED_RESOLUTION_X,
    DESIRED_RESOLUTION_Y,
    DRAWING_DELAY,
    MAX_ITERATION_TIME,
    MAX_TRAINING_ITERATION,
    Point,
    RESOLUTION_SCALE,
    TRAINING_EPOCHS_PER_CALL,
    UPDATE_METRICS_DELAY,
} from "./nn.worker.consts"

import NN, {TrainingDashboard} from "../../neural-network/neural-network";

let neuralNetwork = create_nn(DEFAULT_NN_LAYERS, DEFAULT_LEARNING_RATE);
let points: Point[] = [];
let trainingInputs: matrix.Matrix1D[] = []
let trainingOutputs: matrix.Matrix1D[] = []

let dashboard = createDashboard();

let currentTrainIterations = 0;
let isTraining = false;
let loss = 1;
let accuracy = 0;
let epochsFromLastMetricsUpdate = 0;

let lastDraw = 0;
let lastUpdateMetrics = 0;

function create_nn(sizes: number[], lr: number) {
    const nn = new NN.Models.Sequential(
        new NN.Optimizers.AdamOptimizer({lr, decay: 5e-5, beta1: 0.5}),
        "categoricalCrossEntropy");

    nn.addLayer(new NN.Layers.Dense(2));
    for (const size of sizes) {
        nn.addLayer(new NN.Layers.Dense(size, {
            activation: "relu", weightInitializer: "he",
            options: {
                dropout: 0.1,
                l2WeightRegularization: 5e-5,
                l2BiasRegularization: 5e-5,
            }
        }));
    }

    nn.addLayer(new NN.Layers.Dense(2, {activation: "softmax"}));
    nn.compile();

    return nn;
}

function createDashboard() {
    return new TrainingDashboard(neuralNetwork, trainingInputs, trainingOutputs);
}

addEventListener('message', ({data}) => {
    switch (data.type) {
        case "add_point":
            const point = data.point as Point;

            points.push(point);
            trainingInputs.push([point.x, point.y]);
            trainingOutputs.push(point.type === 0 ? [1, 0] : [0, 1]);

            currentTrainIterations = 0;
            epochsFromLastMetricsUpdate = 0;
            isTraining = true;
            break;

        case "set_points":
            points = data.points as Point[];
            trainingInputs = points.map(point => [point.x, point.y]);
            trainingOutputs = points.map(point => point.type === 0 ? [1, 0] : [0, 1]);
            dashboard = createDashboard();
            currentTrainIterations = 0;
            epochsFromLastMetricsUpdate = 0;
            isTraining = true;
            break;

        case "refresh":
            const newLayersConfig = data.config?.layers || DEFAULT_NN_LAYERS;
            const newLearningRateConfig = data.config?.learningRate || DEFAULT_LEARNING_RATE;

            neuralNetwork = create_nn(newLayersConfig, newLearningRateConfig);
            dashboard = createDashboard();
            currentTrainIterations = 0;
            epochsFromLastMetricsUpdate = 0;
            isTraining = true;
    }
});

function trainBatch() {
    if (!isTraining) {
        return;
    }

    const iterationsLeft = MAX_TRAINING_ITERATION - currentTrainIterations;
    const startTime = performance.now();
    const batchSize = Math.max(64, Math.floor(points.length / 10))

    let iterationsPerCheck = TRAINING_EPOCHS_PER_CALL;
    let epochs;
    for (epochs = 0; epochs < iterationsLeft; epochs++) {
        neuralNetwork.train(trainingInputs, trainingOutputs, {batchSize, progress: false});

        if (epochs % iterationsPerCheck == 0) {
            const trainingTime = performance.now() - startTime;
            if (trainingTime >= MAX_ITERATION_TIME) {
                break;
            }

            if (epochs > 0) {
                // Adaptive calculate optimal iterations count
                iterationsPerCheck = Math.min(1000, Math.max(1, Math.floor(epochs / trainingTime * (MAX_ITERATION_TIME - trainingTime) * 0.9)));
            }
        }
    }

    currentTrainIterations += epochs;
    epochsFromLastMetricsUpdate += epochs;
    const metricsTime = performance.now() - lastUpdateMetrics;
    if (metricsTime >= UPDATE_METRICS_DELAY) {
        const l = neuralNetwork.evaluate(trainingInputs, trainingOutputs);
        loss = l.loss;
        accuracy = l.accuracy;
        isTraining = loss >= DESIRED_LOSS && currentTrainIterations < MAX_TRAINING_ITERATION;

        dashboard.update();
        dashboard.print();

        epochsFromLastMetricsUpdate = 0;
        lastUpdateMetrics = performance.now();
    }

    if (isTraining) {
        sendCurrentState();
    } else {
        lastDraw = 0;
        console.log('*** DATA SET TRAINING FINISHED ***');
        logUtils.print(neuralNetwork, trainingInputs, trainingOutputs);

        sendCurrentState(2);
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
            const typeIndex = (result.indexOf(Math.max(...result))) * Math.max(...result);
            state[y * xSteps + x] = color.getLinearColorBin(COLOR_A_BIN, COLOR_B_BIN, typeIndex);
        }
    }

    const message = {
        type: "training_data",
        epoch: neuralNetwork.epoch,
        loss,
        accuracy,
        isTraining: isTraining,
        state: state.buffer,
        nnSnapshot: neuralNetwork.getSnapshot(),
        width: xSteps,
        height: ySteps
    };

    lastDraw = t;
    postMessage(message, [state.buffer]);
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