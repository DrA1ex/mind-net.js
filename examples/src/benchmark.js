import MindNet, {Matrix} from "mind-net.js";
import * as TimeUtils from "./utils/time.js";

const Sizes = [128, 256, 512];

const model = new MindNet.Models.Sequential();
for (const size of Sizes) {
    model.addLayer(new MindNet.Layers.Dense(size, {activation: "relu"}));
}

model.compile();

const ComputeIters = 10000;
const TrainIters = 5000;
const FullTrainIters = 100;
const Count = 100;

const computeInput = Matrix.random_1d(Sizes[0]);

const trainData = Matrix.random_2d(Count, Sizes[0]);
const trainExcepted = Matrix.random_2d(Count, Sizes[Sizes.length - 1]);

const singleTrainData = trainData.slice(0, 1);
const singleTrainExpected = trainExcepted.slice(0, 1);

for (let i = 0; i < 10; i++) {
    TimeUtils.timeIt(() => model.compute(computeInput), `Compute #${i}`, ComputeIters);
    TimeUtils.timeIt(() => model.train(singleTrainData, singleTrainExpected), `Train (Single) #${i}`, TrainIters);
    TimeUtils.timeIt(() => model.train(trainData, trainExcepted), `Train (Full) #${i}`, FullTrainIters);

    console.log();
}