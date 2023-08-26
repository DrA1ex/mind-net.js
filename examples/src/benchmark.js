import MindNet, {Matrix, ParallelModelWrapper} from "mind-net.js";
import * as TimeUtils from "/Users/akoreshnyak/dev/neural-network/examples/src/utils/time.js";

const Sizes = [128, 256, 256, 512];

const model = new MindNet.Models.Sequential();
for (const size of Sizes) {
    model.addLayer(new MindNet.Layers.Dense(size, {activation: "relu"}));
}

model.compile();

const pModel = new ParallelModelWrapper(model, 5);
await pModel.init();

const ComputeIters = 10000;
const TrainIters = 5
const BatchSize = 128;
const Count = 2000;

const computeInput = Matrix.random_1d(Sizes[0]);

const trainData = Matrix.random_2d(Count, Sizes[0]);
const trainExcepted = Matrix.random_2d(Count, Sizes[Sizes.length - 1]);

const singleTrainData = trainData.slice(0, 1);
const singleTrainExpected = trainExcepted.slice(0, 1);


for (let i = 0; i < 3; i++) {
    await TimeUtils.timeIt(() => pModel.compute(trainData, BatchSize, true), `Worker.Compute (Full) #${i}`, ComputeIters / Count);
    await TimeUtils.timeIt(() => trainData.map(data => model.compute(data)), `Compute (Full) #${i}`, ComputeIters / Count);
    console.log();

    await TimeUtils.timeIt(() => pModel.compute([computeInput]), `Worker.Compute (Single) #${i}`, ComputeIters);
    await TimeUtils.timeIt(() => model.compute(computeInput), `Compute (Single) #${i}`, ComputeIters);
    console.log();

    await TimeUtils.timeIt(() => pModel.train(trainData, trainExcepted, {batchSize: BatchSize}), `Worker.Train (Full) #${i}`, TrainIters);
    await TimeUtils.timeIt(() => model.train(trainData, trainExcepted, {batchSize: BatchSize}), `Train (Full) #${i}`, TrainIters);
    console.log();

    await TimeUtils.timeIt(() => pModel.train(singleTrainData, singleTrainExpected), `Worker.Train (Single) #${i}`, TrainIters * Count);
    await TimeUtils.timeIt(() => model.train(singleTrainData, singleTrainExpected), `Train (Single) #${i}`, TrainIters * Count);
    console.log("\n");
}

await pModel.terminate();