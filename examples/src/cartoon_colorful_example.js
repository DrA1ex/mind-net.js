import tqdm from "tqdm";

import {
    Dense,
    GenerativeAdversarialModel,
    LeakyReluActivation,
    SequentialModel,
    AdamOptimizer,
    TrainingDashboard,
    Iter,
    Matrix,
} from "mind-net.js";

import * as DatasetUtils from "./utils/dataset.js";
import * as ImageUtils from "./utils/image.js";


console.log("Loading data...");

const zipData = await fetch("https://github.com/DrA1ex/mind-net.js/files/12394478/cartoon-500-28.zip")
    .then(r => r.arrayBuffer());

const trainData = (await DatasetUtils.loadDataset(zipData)).splice(0, 500);

console.log("Done.");


const inputDim = 64;
const imageDim = trainData[0].length;

const epochs = 100;
const batchSize = 64;

const lr = 0.01;
const decay = 5e-5;
const beta = 0.5;
const dropout = 0.3;
const loss = "binaryCrossEntropy";
const initializer = "xavier";

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

const generator = new SequentialModel(createOptimizer(), loss);
generator.addLayer(new Dense(inputDim, {
    activation: new LeakyReluActivation({alpha: 0.2}),
    weightInitializer: initializer
}));
generator.addLayer(createHiddenLayer(64));
generator.addLayer(createHiddenLayer(128));
generator.addLayer(new Dense(imageDim, {activation: "tanh", weightInitializer: initializer}));

const discriminator = new SequentialModel(createOptimizer(), loss);
discriminator.addLayer(new Dense(imageDim, {
    activation: new LeakyReluActivation({alpha: 0.2}),
    weightInitializer: initializer
}));
discriminator.addLayer(createHiddenLayer(128));
discriminator.addLayer(createHiddenLayer(64,));
discriminator.addLayer(new Dense(1, {activation: "sigmoid", weightInitializer: initializer}));

const ganModel = new GenerativeAdversarialModel(generator, discriminator, createOptimizer(), loss);

console.log("Training...");

const dashboard = new TrainingDashboard(discriminator, () => {
    const noise = Matrix.random_normal_2d(10, inputDim, -1, 1);
    return noise.map(d => generator.compute(d));
}, Matrix.one_2d(10, 1));

for (const _ of tqdm(Array.from(Iter.range(0, epochs)), {sameLine: true})) {
    console.log("Epoch:", ganModel.ganChain.epoch + 1);
    ganModel.train(trainData, {batchSize});

    dashboard.update();
    dashboard.print();

    await ImageUtils.saveSnapshot(ganModel.generator, ganModel.ganChain.epoch, 10, 3);
    console.log("\n");
}