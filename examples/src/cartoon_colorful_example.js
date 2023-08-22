// Importing necessary modules and libraries
import fs from "fs";
import tqdm from "tqdm";

import {
    Dense,
    GenerativeAdversarialModel,
    LeakyReluActivation,
    SequentialModel,
    AdamOptimizer,
    Iter, ModelSerialization, GanSerialization, Matrix,
} from "mind-net.js";

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
const inputDim = 16;
const imageDim = trainData[0].length;
const gsImageDim = gsTrainData[0].length;
const upscaleImageDim = upscaleTrainData[0].length;
const imageSize = Math.sqrt(imageDim / 3);
const upscaledImageSize = Math.sqrt(upscaleImageDim);

const epochs = 40;
const batchSize = 64;

const lr = 0.005;
const decay = 5e-4;
const beta = 0.5;
const dropout = 0.3;
const loss = "binaryCrossEntropy";
const initializer = "xavier";

// Helper functions and models setup
const createOptimizer = () => new AdamOptimizer({lr, decay, beta1: beta, eps: 1e-7});
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
const generator = new SequentialModel(createOptimizer(), loss);
generator.addLayer(new Dense(inputDim));
generator.addLayer(createHiddenLayer(64));
generator.addLayer(createHiddenLayer(128));
generator.addLayer(new Dense(imageDim, {activation: "tanh", weightInitializer: initializer}));

// Creating the discriminator model
const discriminator = new SequentialModel(createOptimizer(), loss);
discriminator.addLayer(new Dense(imageDim));
discriminator.addLayer(createHiddenLayer(128));
discriminator.addLayer(createHiddenLayer(64));
discriminator.addLayer(new Dense(1, {activation: "sigmoid", weightInitializer: initializer}));

// Creating the generative adversarial (GAN) model
const ganModel = new GenerativeAdversarialModel(generator, discriminator, createOptimizer(), loss);

// Creating the variational autoencoder (VAE) model
const vae = new SequentialModel(createOptimizer(), "mse");
vae.addLayer(new Dense(gsImageDim))
vae.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(128, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(64, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(128, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
vae.addLayer(new Dense(gsImageDim, {activation: "tanh", weightInitializer: initializer}));
vae.compile();

// Creating the Upscaler model
const upscaler = new SequentialModel(createOptimizer(), "mse");
upscaler.addLayer(new Dense(gsImageDim));
upscaler.addLayer(new Dense(256, {activation: "relu", weightInitializer: initializer}));
upscaler.addLayer(new Dense(512, {activation: "relu", weightInitializer: initializer}));
upscaler.addLayer(new Dense(upscaleImageDim, {activation: "tanh", weightInitializer: initializer}));
upscaler.compile();

function _filterWithVAE(input) {
    const gen = generator.compute(input);
    return ModelUtils.processMultiChannelData(vae, gen, 3, gen);
}

function _upscale(input) {
    return ModelUtils.processMultiChannelData(upscaler, _filterWithVAE(input), 3);
}

async function _saveModel() {
    console.log("Saving models...")

    const time = new Date().toISOString();
    const epoch = ganModel.ganChain.epoch;

    if (!fs.existsSync("./out/models")) {
        fs.mkdirSync("./out/models");
    }

    const vaeDump = ModelSerialization.save(vae);
    const vaeFileName = `./out/models/vae_${time}_${epoch}.json`
    fs.writeFileSync(vaeFileName, JSON.stringify(vaeDump));
    console.log(`Saved ${vaeFileName}`);

    const upscalerDump = ModelSerialization.save(upscaler);
    const upscalerFilename = `./out/models/upscaler_${time}_${epoch}.json`
    fs.writeFileSync(upscalerFilename, JSON.stringify(upscalerDump));
    console.log(`Saved ${upscalerFilename}`);

    const ganDump = GanSerialization.save(ganModel);
    const ganFileName = `./out/models/gan_${time}_${epoch}.json`
    fs.writeFileSync(ganFileName, JSON.stringify(ganDump));
    console.log(`Saved ${ganFileName}`);

    const finalGridFileName = `./out/models/sample_${time}_${epoch}.png`;
    await ImageUtils.saveImageGrid(() =>
            _upscale(Matrix.random_normal_1d(inputDim, -1, 1)),
        finalGridFileName, upscaledImageSize, 25, 3, 4, 2);
}

// Save model on exit
process.on("SIGINT", async () => {
    await _saveModel();
    process.exit();
})

console.log("Training...");


// Training loop
for (const _ of tqdm(Array.from(Iter.range(0, epochs)))) {
    console.log("Epoch:", ganModel.ganChain.epoch + 1);

    // Train models
    console.log("GAN train step...");
    ganModel.train(trainData, {batchSize});

    console.log("VAE train step...");
    vae.train(gsTrainData, gsTrainData, {batchSize});

    console.log("Upscaler train step...");
    upscaler.train(gsTrainData, upscaleTrainData, {batchSize});


    console.log("Saving output...");

    // Saving a snapshot of the generator model's output
    await ImageUtils.saveSnapshot(generator, ganModel.ganChain.epoch,
        {label: "generated", channel: 3});

    // Saving an image grid generated by the VAE model
    await ImageUtils.saveImageGrid((x, y) =>
            vae.compute(gsTrainData[(x + y * 10) % trainData.length]),
        `./out/vae_${vae.epoch.toString().padStart(6, "0")}.png`, imageSize, 10, 1);

    // Saving an image grid generated by the upscaler model
    await ImageUtils.saveImageGrid((x, y) =>
            upscaler.compute(gsTrainData[(x + y * 10) % gsTrainData.length]),
        `./out/upscale_${upscaler.epoch.toString().padStart(6, "0")}.png`, upscaledImageSize, 5, 1);

    // Saving an image grid with GAN images filtered with VAE model
    await ImageUtils.saveImageGrid((x, y) =>
            _filterWithVAE(ImageUtils.InputCache.get(generator)[`${x},${y}`]),
        `./out/filtered_${vae.epoch.toString().padStart(6, "0")}.png`, imageSize, 10, 3);

    // Saving an image grid with GAN images filtered with VAE model and upscaled
    await ImageUtils.saveImageGrid((x, y) =>
            _upscale(ImageUtils.InputCache.get(generator)[`${x},${y}`]),
        `./out/filtered_upscaled_${vae.epoch.toString().padStart(6, "0")}.png`, upscaledImageSize, 5, 3, 4, 2);

    console.log("\n");
}

// Save trained models
await _saveModel();