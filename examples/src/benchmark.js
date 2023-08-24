import MindNet, {Matrix} from "mind-net.js";
import * as TimeUtils from "./utils/time.js";

const input = Matrix.random_1d(28 * 28);

const model = new MindNet.Models.Sequential();
model.addLayer(new MindNet.Layers.Dense(28 * 28, {activation: "relu"}));
model.addLayer(new MindNet.Layers.Dense(1024, {activation: "relu"}));
model.addLayer(new MindNet.Layers.Dense(2048, {activation: "relu"}));
model.addLayer(new MindNet.Layers.Dense(64 * 64, {activation: "relu"}));
model.compile();

const Iters = 1000;

for (let i = 0; i < 10; i++) {
    TimeUtils.timeIt(() => model.compute(input), `Compute #${i}`, Iters);
}