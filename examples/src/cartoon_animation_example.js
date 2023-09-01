import fs from "fs";

import {ModelSerialization, GanSerialization, Matrix, ImageUtils, ParallelModelWrapper} from "mind-net.js";
import * as ModelUtils from "./utils/model.js";


const name = "2023-08-30T12:11:05.784Z_100";
const path = "./out/models";
const outPath = "./out/animation";

console.log("Loading models...");

const vaeDump = fs.readFileSync(`${path}/vae_${name}.json`);
const upscalerDump = fs.readFileSync(`${path}/upscaler_${name}.json`);
const ganDump = fs.readFileSync(`${path}/gan_${name}.json`);

const vae = ModelSerialization.load(JSON.parse(vaeDump.toString()));
const upscaler = ModelSerialization.load(JSON.parse(upscalerDump.toString()));
const gan = GanSerialization.load(JSON.parse(ganDump.toString()));

const count = 26;
const framesPerSample = 3;
const scale = 2;

const channels = 3;
const inSize = gan.generator.inputSize;

console.log("Generate images...");

const inputs = new Array((count + 1) * framesPerSample);
const beginning = Matrix.random_normal_1d(inSize, -1, 1);
let start = beginning;
for (let k = 0; k <= count; k++) {
    let next;
    if (k === count) {
        next = beginning;
    } else {
        next = Matrix.random_normal_1d(inSize, -1, 1);
    }

    for (let i = 0; i <= framesPerSample; i++) {
        const data = start.map((v, k) => v + (next[k] - v) * i / framesPerSample);
        inputs[k * framesPerSample + i] = data;
    }

    start = next;
}

const pVae = new ParallelModelWrapper(vae);
const pUpscaler = new ParallelModelWrapper(upscaler);
const pGenerator = new ParallelModelWrapper(gan.generator);

await Promise.all([pVae.init(), pUpscaler.init(), pGenerator.init()]);

const generated = await pGenerator.compute(inputs, {progress: true});
const filtered = await ImageUtils.processMultiChannelDataParallel(pVae, generated, channels);
const upscaled = await ImageUtils.processMultiChannelDataParallel(pUpscaler, filtered, channels);

console.log("Save...");

for (let index = 0; index < upscaled.length; index++) {
    await ModelUtils.saveGeneratedModelsSamples(
        index, outPath, [upscaled[index]],
        {count: 1, channel: channels, scale, prefix: "animation", time: false}
    );
}

await ModelUtils.saveGeneratedModelsSamples(
    "grid", outPath, upscaled,
    {count: Math.floor(Math.sqrt(upscaled.length)), channel: channels, scale}
);

await Promise.all([pVae.terminate(), pUpscaler.terminate(), pGenerator.terminate()]);

console.log("Done!");