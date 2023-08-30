import path from "node:path";

import {
    Dense,
    GenerativeAdversarialModel,
    LeakyReluActivation,
    SequentialModel,
    AdamOptimizer,
    ParallelModelWrapper,
    ParallelGanWrapper,
    Matrix,
    ProgressUtils,
    ImageUtils
} from "mind-net.js";

import * as DatasetUtils from "./utils/dataset.js";
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

const imageChannel = 3;

// Creating grayscale VAE training data from the RGB training data
const gsTrainData = ImageUtils.grayscaleDataset(trainData, imageChannel);

// Creating grayscale Upscaler training data from the big RGB training data
const upscaleTrainData = ImageUtils.grayscaleDataset(bigTrainData);


// Setting up necessary parameters and dimensions
const inputDim = 32;
const imageDim = trainData[0].length;
const gsImageDim = gsTrainData[0].length;
const upscaleImageDim = upscaleTrainData[0].length;

const epochs = 100;
const batchSize = 64;
const epochSamples = 10;
const finalSamples = 20;
const outPath = "./out";

const lr = 0.005;
const decay = 5e-4;
const beta = 0.5;
const dropout = 0.3;
const loss = "binaryCrossEntropy";
const initializer = "xavier";

// Helper functions and models setup
const createOptimizer = (lr) => new AdamOptimizer({lr, decay, beta1: beta, eps: 1e-7});
const createHiddenLayer = (size) => new Dense(size, {
    activation: new LeakyReluActivation({alpha: 0.2}),
    weightInitializer: initializer,
    options: {
        dropout,
        l2WeightRegularization: 1e-4,
        l2BiasRegularization: 1e-4,
    }
});

// Creating the generator model
const generator = new SequentialModel(createOptimizer(lr), loss);
generator.addLayer(new Dense(inputDim));
generator.addLayer(createHiddenLayer(64));
generator.addLayer(createHiddenLayer(128));
generator.addLayer(new Dense(imageDim, {activation: "tanh", weightInitializer: initializer}));

// Creating the discriminator model
const discriminator = new SequentialModel(createOptimizer(lr), loss);
discriminator.addLayer(new Dense(imageDim));
discriminator.addLayer(createHiddenLayer(128));
discriminator.addLayer(createHiddenLayer(64));
discriminator.addLayer(new Dense(1, {activation: "sigmoid", weightInitializer: initializer}));

// Creating the generative adversarial (GAN) model
const ganModel = new GenerativeAdversarialModel(generator, discriminator, createOptimizer(lr), loss);
const pGan = new ParallelGanWrapper(ganModel);

// Creating the variational autoencoder (VAE) model
const vae = new SequentialModel(createOptimizer(lr), "mse");
vae.addLayer(new Dense(gsImageDim))
vae.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(128, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(64, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(128, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(gsImageDim, {activation: "tanh", weightInitializer: initializer}));
vae.compile();

const pVae = new ParallelModelWrapper(vae);

// Creating the Upscaler model
const upscaler = new SequentialModel(createOptimizer(lr), "mse");
upscaler.addLayer(new Dense(gsImageDim));
upscaler.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
upscaler.addLayer(new Dense(512, {activation: "relu", weightInitializer: initializer}));
upscaler.addLayer(new Dense(upscaleImageDim, {activation: "tanh", weightInitializer: initializer}));
upscaler.compile();

const pUpscaler = new ParallelModelWrapper(upscaler);

async function _filterWithVAEBatch(inputs) {
    return ImageUtils.processMultiChannelDataParallel(pVae, inputs, imageChannel);
}

async function _upscaleBatch(inputs) {
    return ImageUtils.processMultiChannelDataParallel(pUpscaler, inputs, imageChannel);
}

async function _saveModel() {
    const savePath = path.join(outPath, "models");
    await ModelUtils.saveModels({vae, upscaler, gan: ganModel}, savePath);

    console.log("Generate final sample set...");

    const generatorInput = Matrix.random_normal_2d(finalSamples ** 2, inputDim, -1, 1);
    const generatedImages = await pGan.compute(generatorInput);
    const vaeFiltered = await _filterWithVAEBatch(generatedImages);
    const genUpscaled = await _upscaleBatch(vaeFiltered);

    await ModelUtils.saveGeneratedModelsSamples("cartoon", savePath, genUpscaled,
        {channel: imageChannel, count: finalSamples});
}

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);

await Promise.all([pVae.init(), pUpscaler.init(), pGan.init()]);

console.log("Training...");

// Training loop
for (const _ of ProgressUtils.progress(epochs)) {
    console.log("Epoch:", ganModel.epoch + 1);

    // Train models
    console.log("Model train step...");

    await Promise.all([
        pGan.train(trainData, {batchSize}),
        pVae.train(gsTrainData, gsTrainData, {batchSize}),
        pUpscaler.train(gsTrainData, upscaleTrainData, {batchSize})
    ]);

    console.log("Saving output...");

    // Saving a snapshot of the generator model's output
    const generatorInput = Matrix.random_normal_2d(epochSamples ** 2, inputDim, -1, 1);
    const generatedImages = await pGan.compute(generatorInput);
    await ModelUtils.saveGeneratedModelsSamples(ganModel.epoch, outPath, generatedImages,
        {channel: imageChannel, count: epochSamples, time: false, prefix: "generated", scale: 4});

    // Saving an image grid generated by the VAE model
    const gsInput = Matrix.fill(i => gsTrainData[i], epochSamples ** 2);
    const vaeImages = await pVae.compute(gsInput);
    await ModelUtils.saveGeneratedModelsSamples(vae.epoch, outPath, vaeImages,
        {channel: 1, count: epochSamples, time: false, prefix: "vae", scale: 4});

    // Saving an image grid generated by the Upscaler model
    const upscalerImages = await pUpscaler.compute(gsInput);
    await ModelUtils.saveGeneratedModelsSamples(upscaler.epoch, outPath, upscalerImages,
        {channel: 1, count: epochSamples, time: false, prefix: "upscaler"});

    // Saving an image grid with GAN images filtered with VAE model
    const vaeFiltered = await _filterWithVAEBatch(generatedImages);
    await ModelUtils.saveGeneratedModelsSamples(vae.epoch, outPath, vaeFiltered,
        {channel: imageChannel, count: epochSamples, time: false, prefix: "vae_filtered", scale: 4});

    // Saving an image grid with GAN images filtered with VAE model and upscaled by Upscaler
    const genUpscaled = await _upscaleBatch(vaeFiltered);
    await ModelUtils.saveGeneratedModelsSamples(vae.epoch, outPath, genUpscaled,
        {channel: imageChannel, count: epochSamples, time: false, prefix: "final"});

    console.log("\n");
    if (quitRequested) break;
}

// Save trained models
await _saveModel();

await Promise.all([pVae.terminate(), pUpscaler.terminate(), pGan.terminate()]);