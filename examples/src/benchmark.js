/**
 *  Install packages first:
 *
 *  npm install @tensorflow/tfjs
 *  npm install brain.js@1.6.0
 *
 */

import MindNet, {Matrix, ParallelModelWrapper, TimeUtils} from "mind-net.js";

import * as tf from '@tensorflow/tfjs';
import brain from "brain.js";

const Sizes = [256, 128, 64, 32, 64, 128, 256];

const model = new MindNet.Models.Sequential();
for (const size of Sizes) {
    model.addLayer(new MindNet.Layers.Dense(size, {activation: "relu"}));
}

model.compile();

const tfModel = tf.sequential();
tfModel.add(tf.layers.dense({units: Sizes[1], inputShape: Sizes[0], activation: "relu"}));
for (const size of Sizes.slice(2)) {
    tfModel.add(tf.layers.dense({units: size, activation: "relu"}));
}

tfModel.compile({loss: "meanSquaredError", optimizer: "sgd"});

const pModel = new ParallelModelWrapper(model, 6);
await pModel.init();

const ComputeIters = 10000;
const TrainIters = 5
const BatchSize = 128;
const Count = 2000;

const trainData = Matrix.random_2d(Count, Sizes[0]);
const singleTrainData = trainData.slice(0, 1);

const tfTrainData = tf.tensor(trainData);
const tfSingleData = tf.tensor(singleTrainData);

const brTrainData = trainData.map(d => ({input: d, output: d}));
const brSingleData = brTrainData.slice(0, 1);

const brModel = new brain.NeuralNetwork({
    hiddenLayers: Sizes.slice(1, Sizes.length - 1),
    activation: 'relu',
});

brModel.train(brSingleData);

const trainOpts = {batchSize: BatchSize, epochs: 1, iterations: 1};

for (let i = 0; i < 3; i++) {
    await TimeUtils.timeIt(() => pModel.compute(trainData, {batchSize: BatchSize}), `Worker.Compute (Full) #${i}`, ComputeIters / Count);
    await TimeUtils.timeIt(() => trainData.map(data => model.compute(data)), `Compute (Full) #${i}`, ComputeIters / Count);
    await TimeUtils.timeIt(() => tfModel.predict(tfTrainData), `TF.Compute (Full) #${i}`, ComputeIters / Count);
    await TimeUtils.timeIt(() => trainData.map(data => brModel.run(data)), `Brain.Compute (Full) #${i}`, ComputeIters / Count);
    console.log();

    await TimeUtils.timeIt(() => pModel.compute(singleTrainData), `Worker.Compute (Single) #${i}`, ComputeIters);
    await TimeUtils.timeIt(() => model.compute(singleTrainData[0]), `Compute (Single) #${i}`, ComputeIters);
    await TimeUtils.timeIt(() => tfModel.predict(tfSingleData), `TF.Compute (Single) #${i}`, ComputeIters);
    await TimeUtils.timeIt(() => brModel.run(singleTrainData[0]), `Brain.Compute (Single) #${i}`, ComputeIters);
    console.log();

    await TimeUtils.timeIt(() => pModel.train(trainData, trainData, trainOpts), `Worker.Train (Full) #${i}`, TrainIters);
    await TimeUtils.timeIt(() => model.train(trainData, trainData, trainOpts), `Train (Full) #${i}`, TrainIters);
    await TimeUtils.timeIt(() => tfModel.fit(tfTrainData, tfTrainData, trainOpts), `TF.Train (Full) #${i}`, TrainIters);
    await TimeUtils.timeIt(() => brModel.train(brTrainData, trainOpts), `Brain.Train (Full) #${i}`, TrainIters);
    console.log();

    await TimeUtils.timeIt(() => pModel.train(singleTrainData, singleTrainData), `Worker.Train (Single) #${i}`, TrainIters * Count);
    await TimeUtils.timeIt(() => model.train(singleTrainData, singleTrainData), `Train (Single) #${i}`, TrainIters * Count);
    await TimeUtils.timeIt(() => tfModel.fit(tfSingleData, tfSingleData, trainOpts), `TF.Train (Single) #${i}`, TrainIters * Count);
    await TimeUtils.timeIt(() => brModel.train(brSingleData, trainOpts), `Brain.Train (Single) #${i}`, TrainIters * Count);
    console.log("\n");
}

await pModel.terminate();