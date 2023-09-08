/**
 *  Install packages first:
 *
 *  npm install @mind-net.js/gpu
 *
 *  Note: override may be needed.
 *  In `package.json` add:
 *  "overrides": {
 *    "gpu.js": {
 *      "gl": "^6.0.2"
 *    }
 *  }
 */

import path from "node:path";

import {
    SequentialModel,
    AdamOptimizer,
    Dense,
    ChainModel,
    ProgressUtils,
    Iter, Matrix, ImageUtils
} from "mind-net.js";

import {GpuModelWrapper} from "@mind-net.js/gpu"

import * as DatasetUtils from "./utils/dataset.js";
import * as ModelUtils from "./utils/model.js";


console.log("Fetching datasets...");

//const DatasetBigUrl = "https://github.com/DrA1ex/mind-net.js/files/12398103/cartoon-2500-64.zip";
//const DatasetBigUrl = "https://github.com/DrA1ex/mind-net.js/files/12456697/mnist-10000-28.zip";
const DatasetBigUrl = "https://github.com/DrA1ex/mind-net.js/files/12407792/cartoon-2500-28.zip";

const zipData = await ProgressUtils.fetchProgress(DatasetBigUrl);

console.log("Preparing...")

const zipLoadingProgress = ProgressUtils.progressCallback({
    update: true,
    color: ProgressUtils.Color.magenta,
    limit: ProgressUtils.ValueLimit.inclusive,
});
const loadedSet = ImageUtils.grayscaleDataset(await DatasetUtils.loadDataset(zipData.buffer, zipLoadingProgress));

// Create variations with different positioning
const setMul = 10;
const dataSet = new Array(loadedSet.length * setMul);
for (let k = 0; k < setMul; k++) {
    for (let i = 0; i < loadedSet.length; i++) {
        const [x, y] = [Math.random() * 2 - 1, Math.random() * 2 - 1];
        dataSet[k * loadedSet.length + i] = ImageUtils.shiftImage(loadedSet[i], x, y, 1);
    }
}

const trainIterations = 100;
const epochsPerIter = 5;
const batchSize = 64;
const imageChannel = 1;
const sampleScale = 4;
const epochSampleSize = 10;
const finalSampleSize = 20;
const outPath = "./out";

const lr = 0.0005;
const decay = 5e-4;
const beta1 = 0.5;
const dropout = 0.3;
const activation = "relu";
const loss = "mse";
const initializer = "xavier";

const LatentSpaceSize = 64;
const Sizes = [512, 256, 128];

const encoder = new SequentialModel(new AdamOptimizer({lr, decay, beta1}), loss);
encoder.addLayer(new Dense(dataSet[0].length));
for (const size of Sizes) {
    encoder.addLayer(
        new Dense(size, {
            activation,
            weightInitializer: initializer,
            options: {dropout}
        })
    );
}
encoder.addLayer(new Dense(LatentSpaceSize, {activation: "sigmoid", weightInitializer: initializer}));

const decoder = new SequentialModel(new AdamOptimizer({lr, decay, beta1}), loss);
decoder.addLayer(new Dense(LatentSpaceSize));
for (const size of Iter.reverse(Sizes)) {
    decoder.addLayer(
        new Dense(size, {
            activation,
            weightInitializer: initializer,
            options: {dropout}
        })
    );
}
decoder.addLayer(new Dense(dataSet[0].length, {weightInitializer: initializer, activation: "tanh"}));

const chain = new ChainModel(new AdamOptimizer({lr, decay, beta1}), loss);
chain.addModel(encoder);
chain.addModel(decoder);

chain.compile();

const pEncoder = new GpuModelWrapper(encoder, {batchSize});
const pDecoder = new GpuModelWrapper(decoder, {batchSize});
const pChain = new GpuModelWrapper(chain, {batchSize});

async function _decode(from, to) {
    const [encFrom, encTo] = await Promise.all([
        pEncoder.compute(from),
        pEncoder.compute(to)
    ]);

    const count = from.length;
    const inputData = new Array(count);
    for (let k = 0; k < count; k++) {
        inputData[k] = encFrom.map(
            (arr, i) => arr.map((value, j) =>
                value + (encTo[i][j] - value) * k / (count - 1)
            )
        );
    }

    return pDecoder.compute(inputData.flat());
}

async function saveModel() {
    const savePath = path.join(outPath, "models");
    await ModelUtils.saveModels({autoencoder: chain}, savePath);

    console.log("Generate final sample set...");

    const genFrom = Matrix.fill(() => dataSet[Math.floor(Math.random() * dataSet.length)], finalSampleSize);
    const genTo = Matrix.fill(() => dataSet[Math.floor(Math.random() * dataSet.length)], finalSampleSize);

    const generated = await _decode(genFrom, genTo);
    await ModelUtils.saveGeneratedModelsSamples("autoencoder", savePath, generated,
        {channel: imageChannel, count: finalSampleSize, scale: sampleScale});
}

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);

const genFrom = Matrix.fill(() => dataSet[Math.floor(Math.random() * dataSet.length)], epochSampleSize);
const genTo = Matrix.fill(() => dataSet[Math.floor(Math.random() * dataSet.length)], epochSampleSize);

console.log("Training...");

for (const epoch of ProgressUtils.progress(trainIterations)) {
    await pChain.train(dataSet, dataSet, {epochs: epochsPerIter});

    const generated = await _decode(genFrom, genTo);
    await ModelUtils.saveGeneratedModelsSamples(epoch, outPath, generated,
        {channel: imageChannel, count: epochSampleSize, scale: sampleScale, time: false, prefix: "autoencoder"});

    if (quitRequested) break;
}

await saveModel();

pChain.destroy();
pEncoder.destroy();
pDecoder.destroy();
