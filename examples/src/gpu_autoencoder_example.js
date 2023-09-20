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

const DatasetSmallUrl = "https://127.0.0.1:8080/datasets/cartoon_avatar_1000_32.zip";
const DatasetBigUrl = "https://127.0.0.1:8080/datasets/cartoon_avatar_1000_64.zip";
const DatasetBiggerUrl = "https://127.0.0.1:8080/datasets/cartoon_avatar_1000_128.zip";
const TestDatasetUrl = "https://127.0.0.1:8080/datasets/doomguy-36-32.zip";


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

let biggerZipData = await ProgressUtils.fetchProgress(DatasetBiggerUrl);
const biggerDataSet = ImageUtils.grayscaleDataset(
    (await DatasetUtils.loadDataset(biggerZipData.buffer, getLoadingProgressFn())).slice(0, CNT)
);
biggerZipData = undefined;

let testZipData = await ProgressUtils.fetchProgress(TestDatasetUrl);
const testColorfulDataSet = await DatasetUtils.loadDataset(testZipData.buffer, getLoadingProgressFn());
testZipData = undefined;

const ShiftingX = 4;
const ShiftingY = 4;

// Create variations with different positioning
const smallDataSetShifted = new Array(smallDataSet.length);
for (let i = 0; i < smallDataSet.length; i++) {
    const x = Math.random() * ShiftingX - ShiftingX / 2
    const y = Math.random() * ShiftingY - ShiftingY / 2;
    smallDataSetShifted[i] = ImageUtils.shiftImage(smallDataSet[i], x, y, 1);
}

const trainIterations = 50;
const epochsPerIterShifter = 10;
const epochsPerIterUpscaler = 5;
const batchSize = 512;
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
const Upscaler2Sizes = [512, 1024];

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
const upscaler2Model = createModel(bigDataSet[0].length, biggerDataSet[0].length, Upscaler2Sizes, lrUpscaler);

const resultingModel = new ChainModel();
resultingModel.addModel(filterModel);
resultingModel.addModel(upscalerModel);
resultingModel.addModel(upscaler2Model);
resultingModel.compile();

const gShifter = new GpuModelWrapper(filterModel, {batchSize});
const gUpscaler = new GpuModelWrapper(upscalerModel, {batchSize: batchSizeUpscaler});
const gUpscaler2 = new GpuModelWrapper(upscaler2Model, {batchSize});
const gResult = new GpuModelWrapper(resultingModel, {batchSize});

const testShiftedDataset = new Array(finalSampleSize ** 2);
for (let i = 0; i < testShiftedDataset.length; i++) {
    const x = Math.round(Math.random() * ShiftingX - ShiftingX / 2);
    const y = Math.round(Math.random() * ShiftingY - ShiftingY / 2);

    const sample = smallDataSet[Math.floor(Math.random() * smallDataSet.length)];
    testShiftedDataset[i] = ImageUtils.shiftImage(sample, x, y, sample[0]);
}

let quitRequested = false;
process.on("SIGINT", async () => quitRequested = true);

async function saveModel() {
    const savePath = path.join(outPath, "models");
    await ModelUtils.saveModels({autoencoder: resultingModel}, savePath);

    console.log("Generate final sample set...");

    const finalGenerated = await ImageUtils.processMultiChannelDataParallel(
        gResult, testColorfulDataSet.slice(0, finalSampleSize ** 2), testImageChannel
    );
    await ModelUtils.saveGeneratedModelsSamples("autoencoder", savePath, finalGenerated,
        {channel: testImageChannel, count: finalSampleSize});
}

console.log("Training...");

const epochTestData = testColorfulDataSet.slice(0, epochSampleSize ** 2);
const gsEpochTestData = Matrix.fill(() => smallDataSetShifted[Math.floor(Math.random() * smallDataSetShifted.length)], epochSampleSize ** 2);

for (const epoch of ProgressUtils.progress(trainIterations)) {
    gShifter.train(smallDataSetShifted, smallDataSet, {epochs: epochsPerIterShifter});
    gUpscaler.train(smallDataSet, bigDataSet, {epochs: epochsPerIterUpscaler});
    gUpscaler2.train(bigDataSet, biggerDataSet, {epochs: epochsPerIterUpscaler});

    const shifterGenerated = gShifter.compute(gsEpochTestData);
    await ModelUtils.saveGeneratedModelsSamples(epoch, outPath, shifterGenerated,
        {
            channel: imageChannel,
            count: epochSampleSize,
            scale: 4,
            time: false,
            prefix: "autoencoder",
            suffix: "shifter"
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

    const upscaler2Generated = gUpscaler2.compute(upscalerGenerated);
    await ModelUtils.saveGeneratedModelsSamples(epoch, outPath, upscaler2Generated,
        {
            channel: imageChannel,
            count: epochSampleSize,
            time: false,
            prefix: "autoencoder",
            suffix: "upscaler2"
        });

    const finalGenerated = await ImageUtils.processMultiChannelDataParallel(gResult, epochTestData, testImageChannel);
    await ModelUtils.saveGeneratedModelsSamples(epoch, outPath, finalGenerated,
        {
            channel: testImageChannel,
            count: epochSampleSize,
            time: false,
            prefix: "autoencoder",
            suffix: "final"
        });

    if (quitRequested) break;
}

await saveModel();

gShifter.destroy();
gUpscaler.destroy();
gResult.destroy();
