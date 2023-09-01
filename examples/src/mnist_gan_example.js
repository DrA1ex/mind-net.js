import path from "node:path";

import {
    Dense,
    GenerativeAdversarialModel,
    LeakyReluActivation,
    SequentialModel,
    AdamOptimizer,
    ParallelGanWrapper,
    Matrix,
    ImageUtils,
    ProgressUtils,
} from "mind-net.js";

import * as DatasetUtils from "./utils/dataset.js";
import * as ModelUtils from "./utils/model.js";

const DatasetUrl = "https://github.com/DrA1ex/mind-net.js/files/12456697/mnist-10000-28.zip";

console.log("Fetching datasets...");

// Fetching the dataset zip file and converting it to an ArrayBuffer
const zipData = await ProgressUtils.fetchProgress(DatasetUrl);

console.log("Preparing...");

// Loading datasets from the zip files
const trainData = ImageUtils.grayscaleDataset(
    await DatasetUtils.loadDataset(zipData.buffer)
);

// You can reduce dataset length
const CNT = 2500;
trainData.splice(CNT);

// Setting up necessary parameters and dimensions
const inputDim = 64;
const imageDim = trainData[0].length;

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

const createOptimizer = (lr) => new AdamOptimizer({lr, decay, beta1: beta, eps: 1e-7});
const createHiddenLayer = (size) => new Dense(size, {
    activation: new LeakyReluActivation({alpha: 0.2}),
    weightInitializer: initializer,
    options: {dropout}
});

const layerSizes = [128, 256, 512];

//Creating the generator model
const generator = new SequentialModel(createOptimizer(lr), loss);
generator.addLayer(new Dense(inputDim));
for (const layerSize of layerSizes) generator.addLayer(createHiddenLayer(layerSize));
generator.addLayer(new Dense(imageDim, {activation: "tanh", weightInitializer: initializer}));
generator.compile();

// Creating the discriminator model
const discriminator = new SequentialModel(createOptimizer(lr), loss);
discriminator.addLayer(new Dense(imageDim));
for (const layerSize of layerSizes.reverse()) discriminator.addLayer(createHiddenLayer(layerSize));
discriminator.addLayer(new Dense(1, {activation: "sigmoid", weightInitializer: initializer}));
discriminator.compile();


//Creating the generative adversarial (GAN) model
const ganModel = new GenerativeAdversarialModel(generator, discriminator, createOptimizer(lr), loss);

const pGan = new ParallelGanWrapper(ganModel, 8);
await pGan.init();

async function _saveModel() {
    const savePath = path.join(outPath, "models");
    await ModelUtils.saveModels({mnist_gan: ganModel}, savePath);

    console.log("Generate final sample set...");

    const generatorInput = Matrix.random_normal_2d(finalSamples ** 2, inputDim, -1, 1);
    const generatedImages = await pGan.compute(generatorInput);

    await ModelUtils.saveGeneratedModelsSamples("mnist", savePath, generatedImages,
        {count: finalSamples});
}

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);

console.log("Training...");

const generatorInput = Matrix.random_normal_2d(epochSamples ** 2, inputDim, -1, 1);

// Training loop
for (const _ of ProgressUtils.progress(epochs)) {
    console.log("Epoch:", ganModel.epoch + 1);

    await pGan.train(trainData, {batchSize});

    console.log("Saving output...");
    const generatedImages = await pGan.compute(generatorInput);
    await ModelUtils.saveGeneratedModelsSamples(ganModel.epoch, outPath, generatedImages,
        {count: epochSamples, time: false, prefix: "generated", scale: 4});

    console.log("\n");
    if (quitRequested) break;
}

// Save trained models
await _saveModel();

await pGan.terminate();