// Importing necessary modules and libraries
import tqdm from "tqdm";

import {SequentialModel, AdamOptimizer, Dense, Iter, Matrix, ParallelModelWrapper,} from "mind-net.js";

import * as DatasetUtils from "./utils/dataset.js";
import * as ImageUtils from "./utils/image.js";
import * as ModelUtils from "./utils/model.js";


const DatasetUrl = "https://github.com/DrA1ex/mind-net.js/files/12407792/cartoon-2500-28.zip";
const DatasetBigUrl = "https://github.com/DrA1ex/mind-net.js/files/12398103/cartoon-2500-64.zip";

console.log("Fetching datasets...");

// Fetching the dataset zip file and converting it to an ArrayBuffer
const [zipData, zipBigData] = await Promise.all([
    fetch(DatasetUrl).then(r => r.arrayBuffer()),
    fetch(DatasetBigUrl).then(r => r.arrayBuffer())
]);

console.log("Preparing...");

// Loading datasets from the zip files
const [trainData, bigTrainData] = await Promise.all([
    DatasetUtils.loadDataset(zipData),
    DatasetUtils.loadDataset(zipBigData)
]);

// You can reduce dataset length
const CNT = 2500;
trainData.splice(CNT);
bigTrainData.splice(CNT);

// Creating grayscale VAE training data from the RGB training data
const gsTrainData = ImageUtils.grayscaleDataset(trainData, 3);

// Creating grayscale Upscaler training data from the big RGB training data
const upscaleTrainData = ImageUtils.grayscaleDataset(bigTrainData);

// Setting up necessary parameters and dimensions
const lr = 0.005;
const decay = 5e-4;
const beta = 0.5;
const loss = "mse";
const initializer = "xavier";

const epochs = 40;
const batchSize = 64;
const upscalerChunksDivider = 4;
const imageChannel = 3;
const epochSampleSize = 10;
const finalSampleSize = 20;
const upscalerChunks = upscalerChunksDivider * upscalerChunksDivider;

const imageDim = trainData[0].length;
const gsImageDim = gsTrainData[0].length;
const upscaleImageDim = upscaleTrainData[0].length;
const imageSize = Math.sqrt(imageDim / imageChannel);
const upscaledImageSize = Math.sqrt(upscaleImageDim);

// Create chunked training set
const upscaleInChunks = gsTrainData.map(
    entry => ImageUtils.splitChunks(entry, imageSize, imageSize / upscalerChunksDivider)
).flat();

const upscaleOutChunks = upscaleTrainData.map(
    entry => ImageUtils.splitChunks(entry, upscaledImageSize, upscaledImageSize / upscalerChunksDivider)
).flat();

// Helper functions and models setup
const createOptimizer = (lr) => new AdamOptimizer({lr, decay, beta1: beta, eps: 1e-7});
const createHiddenLayer = (size) => new Dense(size, {activation: "relu", weightInitializer: initializer});

// Creating the Upscaler model
const upscaler = new SequentialModel(createOptimizer(lr), loss);
upscaler.addLayer(new Dense(gsImageDim / upscalerChunks));
upscaler.addLayer(createHiddenLayer(64));
upscaler.addLayer(createHiddenLayer(128));
upscaler.addLayer(new Dense(upscaleImageDim / upscalerChunks, {activation: "tanh", weightInitializer: initializer}));
upscaler.compile();

async function _upscaleBatch(inputs) {
    return ImageUtils.processMultiChunkDataParallel(pUpscaler, inputs, imageChannel);
}

async function _saveModel() {
    const models = {chunked_upscaler: upscaler};
    const outPath = "./out/models/";

    await ModelUtils.saveModels(models, outPath);

    const generated = await _upscaleBatch(Matrix.fill(
        () => trainData[Math.floor(Math.random() * trainData.length)],
        finalSampleSize * finalSampleSize
    ));

    await ModelUtils.saveGeneratedModelsSamples("chunked_upscaler", outPath, generated,
        {channel: imageChannel, count: finalSampleSize});
}

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);

const pUpscaler = new ParallelModelWrapper(upscaler, 6);
await pUpscaler.init();

console.log("Training...");

// Training loop
for (const _ of tqdm(Array.from(Iter.range(0, epochs)))) {
    console.log("Epoch:", upscaler.epoch + 1);

    // Train models
    await pUpscaler.train(upscaleInChunks, upscaleOutChunks, {batchSize});

    console.log("Saving output...");

    const inputs = Matrix.fill(() => trainData[Math.floor(Math.random() * trainData.length)],
        epochSampleSize * epochSampleSize);
    const generated = await _upscaleBatch(inputs);

    // Saving an image grid with GAN images filtered with VAE model and upscaled
    await ModelUtils.saveGeneratedModelsSamples(
        upscaler.epoch, "./out", generated,
        {count: epochSampleSize, channel: imageChannel, time: false, prefix: "chunked_upscaler"}
    );

    console.log("\n");
    if (quitRequested) break;
}

// Save trained models
await _saveModel();
await pUpscaler.terminate();