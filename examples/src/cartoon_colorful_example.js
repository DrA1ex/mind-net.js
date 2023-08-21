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
import {grayscaleDataset} from "./utils/image.js";


//const DatasetUrl = "https://github.com/DrA1ex/mind-net.js/files/12394478/cartoon-500-28.zip";
const DatasetUrl = "https://github.com/DrA1ex/mind-net.js/files/12396106/cartoon-2500-28.zip";

console.log("Fetching dataset...");

// Fetching the dataset zip file and converting it to an ArrayBuffer
const zipData = await fetch(DatasetUrl).then(r => r.arrayBuffer());


console.log("Preparing...");

// Loading the dataset from the zip file
const trainData = await DatasetUtils.loadDataset(zipData);

// Creating grayscale VAE training data from the RGB training data
const gsTrainData = grayscaleDataset(trainData, 3);


// Setting up necessary parameters and dimensions
const inputDim = 16;
const imageDim = trainData[0].length;
const gsImageDim = gsTrainData[0].length;
const imageSize = Math.sqrt(trainData[0].length / 3);

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

// Creating the generative adversarial model
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

function _filterWithVAE(input) {
    const gen = generator.compute(input);
    const channelData = new Array(gsImageDim);
    for (let c = 0; c < 3; c++) {
        ImageUtils.getChannel(gen, c, 3, channelData);
        const filtered = vae.compute(channelData);
        ImageUtils.setChannel(gen, filtered, c, 3);
    }
    return gen;
}

async function _saveModel() {
    console.log("Saving models...")

    const time = new Date().toISOString();

    const vaeDump = ModelSerialization.save(vae);
    const vaeFileName = `./out/vae_${time}_${vae.epoch}.json`
    fs.writeFileSync(vaeFileName, JSON.stringify(vaeDump));
    console.log(`Saved ${vaeFileName}`);

    const ganDump = GanSerialization.save(ganModel);
    const ganFileName = `./out/gan_${time}_${ganModel.ganChain.epoch}.json`
    fs.writeFileSync(ganFileName, JSON.stringify(ganDump));
    console.log(`Saved ${ganFileName}`);

    const finalGridFileName = `./out/final_${time}.png`;
    await ImageUtils.saveImageGrid(() => _filterWithVAE(Matrix.random_1d(inputDim)),
        finalGridFileName, imageSize, 25, 3);
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

    // Train GAN and VAE models
    ganModel.train(trainData, {batchSize});
    vae.train(gsTrainData, gsTrainData, {batchSize});

    // Saving a snapshot of the generator model's output
    await ImageUtils.saveSnapshot(generator, ganModel.ganChain.epoch,
        {label: "generated", channel: 3});

    // Saving an image grid generated by the VAE model
    await ImageUtils.saveImageGrid((x, y) =>
            vae.compute(gsTrainData[(x + y * 10) % trainData.length]),
        `./out/vae_${vae.epoch.toString().padStart(6, "0")}.png`, imageSize, 10, 1);

    // Saving an image grid with GAN images filtered with VAE model
    await ImageUtils.saveImageGrid((x, y) =>
            _filterWithVAE(ImageUtils.InputCache.get(generator)[`${x},${y}`]),
        `./out/filtered_${vae.epoch.toString().padStart(6, "0")}.png`, imageSize, 10, 3);

    console.log("\n");
}

// Save trained models
await _saveModel();