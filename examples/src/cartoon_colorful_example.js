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
    ImageUtils,
    ColorUtils
} from "mind-net.js";

import {GpuModelWrapper, GpuGanWrapper} from "@mind-net.js/gpu"

import * as DatasetUtils from "./utils/dataset.js";
import * as ModelUtils from "./utils/model.js";


const DatasetUrl = "https://127.0.0.1:8080/datasets/cartoon_avatar_4000_32.zip";
const DatasetBigUrl = "https://127.0.0.1:8080/datasets/cartoon_avatar_4000_64.zip";

console.log("Fetching datasets...");

// Fetching the dataset zip file and converting it to an ArrayBuffer
const zipData = await ProgressUtils.fetchProgress(DatasetUrl);
const zipBigData = await ProgressUtils.fetchProgress(DatasetBigUrl);

console.log("Preparing...");

// Loading datasets from the zip files
const [trainData, bigTrainData] = await Promise.all([
    DatasetUtils.loadDataset(zipData.buffer),
    DatasetUtils.loadDataset(zipBigData.buffer)
]);

// You can reduce dataset length
const CNT = 4000;
trainData.splice(CNT);
bigTrainData.splice(CNT);

const imageChannel = 3;

// Creating grayscale AE training data from the RGB training data
const gsTrainData = ImageUtils.grayscaleDataset(trainData, imageChannel);

// Creating grayscale Upscaler training data from the big RGB training data
const upscaleTrainData = ImageUtils.grayscaleDataset(bigTrainData);

for (const tData of trainData) {
    ColorUtils.transformColorSpace(ColorUtils.tanhToRgb, tData, imageChannel, tData);
    ColorUtils.transformColorSpace(ColorUtils.rgbToLab, tData, imageChannel, tData);
    ColorUtils.transformColorSpace(ColorUtils.labToTanh, tData, imageChannel, tData);
}

// Setting up necessary parameters and dimensions
const inputDim = 64;
const imageDim = trainData[0].length;
const gsImageDim = gsTrainData[0].length;
const upscaleImageDim = upscaleTrainData[0].length;

const epochs = 100;
const batchSize = 128;
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
const createHiddenLayer = (size, activation = undefined) => new Dense(size, {
    activation: activation ?? new LeakyReluActivation({alpha: 0.2}),
    weightInitializer: initializer,
    options: {
        dropout,
        l2WeightRegularization: 1e-4,
        l2BiasRegularization: 1e-4,
    }
});

// Creating the generator model
const generator = new SequentialModel(createOptimizer(lr * 1.2), loss);
generator.addLayer(new Dense(inputDim));
generator.addLayer(createHiddenLayer(128, "relu"));
generator.addLayer(createHiddenLayer(256, "relu"));
generator.addLayer(new Dense(imageDim, {activation: "tanh", weightInitializer: initializer}));

// Creating the discriminator model
const discriminator = new SequentialModel(createOptimizer(lr), loss);
discriminator.addLayer(new Dense(imageDim));
discriminator.addLayer(createHiddenLayer(256));
discriminator.addLayer(createHiddenLayer(128));
discriminator.addLayer(new Dense(1, {activation: "sigmoid", weightInitializer: initializer}));

// Creating the generative adversarial (GAN) model
const ganModel = new GenerativeAdversarialModel(generator, discriminator, createOptimizer(lr), loss);
const pGan = new GpuGanWrapper(ganModel, {batchSize});

// Creating the variational autoencoder (AE) model
const ae = new SequentialModel(createOptimizer(lr), "mse");
ae.addLayer(new Dense(gsImageDim))
ae.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
ae.addLayer(new Dense(128, {activation: "relu", weightInitializer: initializer}));
ae.addLayer(new Dense(64, {activation: "relu", weightInitializer: initializer}));
ae.addLayer(new Dense(128, {activation: "relu", weightInitializer: initializer}));
ae.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
ae.addLayer(new Dense(gsImageDim, {activation: "tanh", weightInitializer: initializer}));
ae.compile();

const pAe = new GpuModelWrapper(ae, {batchSize});

// Creating the Upscaler model
const upscaler = new SequentialModel(createOptimizer(lr), "mse");
upscaler.addLayer(new Dense(gsImageDim));
upscaler.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
upscaler.addLayer(new Dense(512, {activation: "relu", weightInitializer: initializer}));
upscaler.addLayer(new Dense(upscaleImageDim, {activation: "tanh", weightInitializer: initializer}));
upscaler.compile();

const pUpscaler = new GpuModelWrapper(upscaler, {batchSize});

async function _filterWithAEBatch(inputs) {
    return ImageUtils.processMultiChannelDataParallel(pAe, inputs, imageChannel);
}

async function _upscaleBatch(inputs) {
    return ImageUtils.processMultiChannelDataParallel(pUpscaler, inputs, imageChannel);
}

async function _saveModel() {
    const savePath = path.join(outPath, "models");
    await ModelUtils.saveModels({ae, upscaler, gan: ganModel}, savePath);

    console.log("Generate final sample set...");

    const generatorInput = Matrix.random_normal_2d(finalSamples ** 2, inputDim, -1, 1);
    const generatedImages = await pGan.compute(generatorInput);
    const aeFiltered = await _filterWithAEBatch(generatedImages);
    const genUpscaled = await _upscaleBatch(aeFiltered);

    await ModelUtils.saveGeneratedModelsSamples("cartoon", savePath, genUpscaled,
        {channel: imageChannel, count: finalSamples});
}

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);

//await Promise.all([pAe.init(), pUpscaler.init(), pGan.init()]);

console.log("Training...");

const generatorInput = Matrix.random_normal_2d(epochSamples ** 2, inputDim, -1, 1);

// Training loop
for (const _ of ProgressUtils.progress(epochs)) {
    console.log("Epoch:", ganModel.epoch + 1);

    // Train models
    console.log("Model train step...");

    await pGan.train(trainData, {batchSize});
    await pAe.train(gsTrainData, gsTrainData, {batchSize});
    await pUpscaler.train(gsTrainData, upscaleTrainData, {batchSize});

    console.log("Saving output...");

    // Saving a snapshot of the generator model's output
    const generatedImages = await pGan.compute(generatorInput);
    for (const tData of generatedImages) {
        ColorUtils.transformColorSpace(ColorUtils.tanhToLab, tData, imageChannel, tData);
        ColorUtils.transformColorSpace(ColorUtils.labToRgb, tData, imageChannel, tData);
        ColorUtils.transformColorSpace(ColorUtils.rgbToTanh, tData, imageChannel, tData);
    }

    await ModelUtils.saveGeneratedModelsSamples(ganModel.epoch, outPath, generatedImages,
        {channel: imageChannel, count: epochSamples, time: false, prefix: "generated", scale: 4});

    // Saving an image grid generated by the AE model
    const gsInput = Matrix.fill(i => gsTrainData[i], epochSamples ** 2);
    const aeImages = await pAe.compute(gsInput);
    await ModelUtils.saveGeneratedModelsSamples(ae.epoch, outPath, aeImages,
        {channel: 1, count: epochSamples, time: false, prefix: "ae", scale: 4});

    // Saving an image grid generated by the Upscaler model
    const upscalerImages = await pUpscaler.compute(gsInput);
    await ModelUtils.saveGeneratedModelsSamples(upscaler.epoch, outPath, upscalerImages,
        {channel: 1, count: epochSamples, time: false, prefix: "upscaler"});

    // Saving an image grid with GAN images filtered with AE model
    const aeFiltered = await _filterWithAEBatch(generatedImages);
    await ModelUtils.saveGeneratedModelsSamples(ae.epoch, outPath, aeFiltered,
        {channel: imageChannel, count: epochSamples, time: false, prefix: "ae_filtered", scale: 4});

    // Saving an image grid with GAN images filtered with AE model and upscaled by Upscaler
    const genUpscaled = await _upscaleBatch(aeFiltered);
    await ModelUtils.saveGeneratedModelsSamples(ae.epoch, outPath, genUpscaled,
        {channel: imageChannel, count: epochSamples, time: false, prefix: "final"});

    console.log("\n");
    if (quitRequested) break;
}

// Save trained models
await _saveModel();

//await Promise.all([pAe.terminate(), pUpscaler.terminate(), pGan.terminate()]);