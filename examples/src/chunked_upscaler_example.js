// Importing necessary modules and libraries
import tqdm from "tqdm";

import {SequentialModel, AdamOptimizer, Dense, Iter,} from "mind-net.js";

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
const upscalerChunks = upscalerChunksDivider * upscalerChunksDivider;

const imageDim = trainData[0].length;
const gsImageDim = gsTrainData[0].length;
const upscaleImageDim = upscaleTrainData[0].length;
const imageSize = Math.sqrt(imageDim / 3);
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

function _upscale(input) {
    const result = new Array(upscaleImageDim * 3);

    for (let c = 0; c < 3; c++) {
        const channel = ImageUtils.getChannel(input, c, 3);
        const chunks = ImageUtils.splitChunks(channel, imageSize, imageSize / upscalerChunksDivider);
        const outChunks = chunks.map(chunk => upscaler.compute(chunk));
        const upscaled = ImageUtils.joinChunks(outChunks);
        ImageUtils.setChannel(result, upscaled, c, 3);
    }

    return result;
}

async function _saveModel() {
    const models = {chunked_upscaler: upscaler};
    const outPath = "./out/models/";

    await ModelUtils.saveModels(models, outPath);
    await ModelUtils.saveModelsSamples("chunked_upscaler", outPath, upscaledImageSize,
        () => _upscale(trainData[Math.floor(Math.random() * trainData.length)]),
        {channel: 3, count: 10});
}

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);


console.log("Training...");


// Training loop
for (const _ of tqdm(Array.from(Iter.range(0, epochs)))) {
    console.log("Epoch:", upscaler.epoch + 1);

    // Train models
    upscaler.train(upscaleInChunks, upscaleOutChunks, {batchSize});

    console.log("Saving output...");

    // Saving an image grid with GAN images filtered with VAE model and upscaled
    await ImageUtils.saveImageGrid(() =>
            _upscale(trainData[Math.floor(Math.random() * trainData.length)]),
        `./out/chunked_upscaler_${upscaler.epoch.toString().padStart(6, "0")}.png`,
        upscaledImageSize, 5, 3, 4, 2);

    console.log("\n");
    if (quitRequested) break;
}

// Save trained models
await _saveModel();