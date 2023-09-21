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
    ImageUtils,
    Matrix
} from "mind-net.js";

import {GpuModelWrapper} from "@mind-net.js/gpu"

import * as DatasetUtils from "./utils/dataset.js";
import * as ModelUtils from "./utils/model.js";


console.log("Fetching datasets...");

const DatasetSmallUrl = "https://github.com/DrA1ex/mind-net.js/files/12684744/cartoon_avatar_1000_32.zip";
const DatasetBigUrl = "https://github.com/DrA1ex/mind-net.js/files/12684745/cartoon_avatar_1000_64.zip";
const TestDatasetUrl = "https://github.com/DrA1ex/mind-net.js/files/12684746/face-25-32.zip";

console.log("Preparing...")

const getLoadingProgressFn = () => ProgressUtils.progressCallback({
    update: true,
    color: ProgressUtils.Color.magenta,
    limit: ProgressUtils.ValueLimit.inclusive,
});

const CNT = 2500;

let smallZipData = await ProgressUtils.fetchProgress(DatasetSmallUrl);
const smallDataSet = ImageUtils.grayscaleDataset(
    (await DatasetUtils.loadDataset(smallZipData.buffer, getLoadingProgressFn())).slice(0, CNT)
);
smallZipData = undefined;

let bigZipData = await ProgressUtils.fetchProgress(DatasetBigUrl);
const bigDataSet = ImageUtils.grayscaleDataset(
    (await DatasetUtils.loadDataset(bigZipData.buffer, getLoadingProgressFn())).slice(0, CNT)
);
bigZipData = undefined;

let testZipData = await ProgressUtils.fetchProgress(TestDatasetUrl);
const testColorfulDataSet = await DatasetUtils.loadDataset(testZipData.buffer, getLoadingProgressFn());
testZipData = undefined;

const trainIterations = 50;
const epochsPerIterFilter = 10;
const epochsPerIterUpscaler = 5;
const batchSize = 256;
const batchSizeUpscaler = 512;
const imageChannel = 1;
const testImageChannel = 3;
const epochSampleSize = Math.min(10, Math.floor(Math.sqrt(testColorfulDataSet.length)));
const finalSampleSize = Math.min(20, Math.floor(Math.sqrt(testColorfulDataSet.length)));
const outPath = "./out";

const lr = 0.0003;
const lrUpscaler = 0.0005;
const decay = 5e-4;
const beta1 = 0.5;
const dropout = 0.2;
const activation = "relu";
const loss = "l2";
const initializer = "xavier";

const FilterSizes = [512, 384, 256, 384, 512];
const UpscalerSizes = [512, 640, 784, 896, 1024];

function createOptimizer(lr) {
    return new AdamOptimizer({lr, decay, beta1});
}

function createHiddenLayer(size) {
    return new Dense(size, {
        activation,
        weightInitializer: initializer,
        options: {dropout}
    });
}

function createModel(inSize, outSize, hiddenLayers, lr) {
    const model = new SequentialModel(createOptimizer(lr), loss);
    model.addLayer(new Dense(inSize));
    for (const size of hiddenLayers) model.addLayer(createHiddenLayer(size));
    model.addLayer(new Dense(outSize, {activation: "tanh"}));
    model.compile();

    return model;
}

const filterModel = createModel(smallDataSet[0].length, smallDataSet[0].length, FilterSizes, lr);
const upscalerModel = createModel(smallDataSet[0].length, bigDataSet[0].length, UpscalerSizes, lrUpscaler);

const resultingModel = new ChainModel();
resultingModel.addModel(filterModel);
resultingModel.addModel(upscalerModel);
resultingModel.compile();

const gFilter = new GpuModelWrapper(filterModel, {batchSize});
const gUpscaler = new GpuModelWrapper(upscalerModel, {batchSize: batchSizeUpscaler});
const gResult = new GpuModelWrapper(resultingModel, {batchSize});

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);

async function saveModel() {
    const savePath = path.join(outPath, "models");
    await ModelUtils.saveModels({autoencoder: resultingModel}, savePath, true);

    console.log("Generate final sample set...");

    const finalGenerated = await ImageUtils.processMultiChannelDataParallel(
        gResult, testColorfulDataSet.slice(0, finalSampleSize ** 2), testImageChannel
    );

    await ModelUtils.saveGeneratedModelsSamples("autoencoder", savePath, finalGenerated,
        {channel: testImageChannel, scale: 2, count: finalSampleSize});
}

console.log("Training...");

const epochTestData = testColorfulDataSet.slice(0, epochSampleSize ** 2);
const gsEpochTestData = Matrix.fill(() => smallDataSet[Math.floor(Math.random() * smallDataSet.length)], epochSampleSize ** 2);

for (const epoch of ProgressUtils.progress(trainIterations)) {
    gFilter.train(smallDataSet, smallDataSet, {epochs: epochsPerIterFilter});
    gUpscaler.train(smallDataSet, bigDataSet, {epochs: epochsPerIterUpscaler});

    const filter = gFilter.compute(gsEpochTestData);
    await ModelUtils.saveGeneratedModelsSamples(epoch, outPath, filter,
        {
            channel: imageChannel,
            count: epochSampleSize,
            scale: 4,
            time: false,
            prefix: "autoencoder",
            suffix: "filter"
        });

    const upscalerGenerated = gUpscaler.compute(gsEpochTestData);
    await ModelUtils.saveGeneratedModelsSamples(epoch, outPath, upscalerGenerated,
        {
            channel: imageChannel,
            count: epochSampleSize,
            scale: 2,
            time: false,
            prefix: "autoencoder",
            suffix: "upscaler"
        });

    const finalGenerated = await ImageUtils.processMultiChannelDataParallel(gResult, epochTestData, testImageChannel);
    await ModelUtils.saveGeneratedModelsSamples(epoch, outPath, finalGenerated,
        {
            channel: testImageChannel,
            count: epochSampleSize,
            scale: 2,
            time: false,
            prefix: "autoencoder",
            suffix: "final",
        });

    if (quitRequested) break;
}

await saveModel();

gFilter.destroy();
gUpscaler.destroy();
gResult.destroy();
