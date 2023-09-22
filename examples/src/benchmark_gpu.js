/**
 *  Install packages first:
 *
 *  npm install @mind-net.js/gpu
 *  npm install @tensorflow/tfjs-node-gpu
 *  npm install brain.js
 *
 *  Note: override may be needed.
 *  In `package.json` add:
 *  "overrides": {
 *    "gpu.js": {
 *      "gl": "^6.0.2"
 *    }
 *  }
 */

import MindNet, {Matrix, TimeUtils} from "mind-net.js";

import * as tf from '@tensorflow/tfjs-node-gpu';
import brain from "brain.js";
import {GpuModelWrapper} from "@mind-net.js/gpu";

const Sizes = [512, 256, 128, 64, 128, 256, 512];
const ComputeIters = 20000;
const TrainIters = 10;
const BatchSize = 128;
const Count = 2000;

console.log("Preparing...");

const model = new MindNet.Models.Sequential();
for (const size of Sizes) {
    model.addLayer(new MindNet.Layers.Dense(size, {activation: "relu"}));
}

model.compile();

const gpuModel = new GpuModelWrapper(model, {batchSize: BatchSize});

const tfModel = tf.sequential();
tfModel.add(tf.layers.dense({units: Sizes[1], inputShape: Sizes[0], activation: "relu"}));
for (const size of Sizes.slice(2)) {
    tfModel.add(tf.layers.dense({units: size, activation: "relu"}));
}

tfModel.compile({loss: "meanSquaredError", optimizer: "sgd"});

const trainData = Matrix.random_2d(Count, Sizes[0]);

const tfTrainData = tf.tensor(trainData.map(t => Array.from(t)));

const brTrainData = trainData.map(d => ({input: d, output: d}));

const brModel = new brain.NeuralNetworkGPU({
    hiddenLayers: Sizes.slice(1, Sizes.length - 1),
    activation: 'relu',
    mode: "gpu"
});

brModel.train(brTrainData.slice(0, 1));

const trainOpts = {batchSize: BatchSize, epochs: 1, iterations: 1, progress: false, verbose: false};

console.log("Testing...\n");

for (let i = 0; i < 3; i++) {
    await TimeUtils.timeIt(() => gpuModel.compute(trainData), `GPU.Compute (Full) #${i}`, ComputeIters / Count);
    await TimeUtils.timeIt(() => gpuModel.train(trainData, trainData, trainOpts), `GPU.Train (Full) #${i}`, TrainIters);
}
console.log();

for (let i = 0; i < 3; i++) {
    await TimeUtils.timeIt(() => tfModel.predict(tfTrainData), `TF.Compute (Full) #${i}`, ComputeIters / Count);
    await TimeUtils.timeIt(() => tfModel.fit(tfTrainData, tfTrainData, trainOpts), `TF.Train (Full) #${i}`, TrainIters);
}
console.log();

for (let i = 0; i < 3; i++) {
    await TimeUtils.timeIt(() => trainData.map(data => brModel.run(data)), `Brain.Compute (Full) #${i}`, ComputeIters / Count);
    await TimeUtils.timeIt(() => brModel.train(brTrainData, trainOpts), `Brain.Train (Full) #${i}`, TrainIters);
}
console.log();

gpuModel.destroy();