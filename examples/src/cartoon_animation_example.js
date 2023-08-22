import fs from "fs";
import {ModelSerialization, GanSerialization, Matrix,} from "mind-net.js";
import * as ImageUtils from "./utils/image.js";
import * as ModelUtils from "./utils/model.js";


const name = "2023-08-22T12:45:42.929Z_40";
const path = "./out/models";

const vaeDump = fs.readFileSync(`${path}/vae_${name}.json`);
const upscalerDump = fs.readFileSync(`${path}/upscaler_${name}.json`);
const ganDump = fs.readFileSync(`${path}/gan_${name}.json`);

const vae = ModelSerialization.load(JSON.parse(vaeDump.toString()));
const upscaler = ModelSerialization.load(JSON.parse(upscalerDump.toString()));
const gan = GanSerialization.load(JSON.parse(ganDump.toString()));

const count = 10;
const framesPerSample = 6;
const scale = 2;

const inSize = gan.generator.layers[0].size;
const outSize = Math.sqrt(upscaler.layers[upscaler.layers.length - 1].size);

const beginning = Matrix.random_normal_1d(inSize, -1, 1);
let start = beginning;
for (let k = 0; k <= count; k++) {
    let next;
    if (k === 10) {
        next = beginning;
    } else {
        next = Matrix.random_normal_1d(inSize, -1, 1);
    }

    for (let i = 0; i <= framesPerSample; i++) {
        const data = start.map((v, k) => v + (next[k] - v) * i / framesPerSample);

        await ImageUtils.saveImageGrid(() => {
                const gen = gan.generator.compute(data);
                const filtered = ModelUtils.processMultiChannelData(vae, gen, 3, gen);
                return ModelUtils.processMultiChannelData(upscaler, filtered, 3);
            },
            `./out/animation/animation_${(k * framesPerSample + i).toString().padStart(6, "0")}.png`,
            outSize, 1, 3, 0, scale
        );
    }

    start = next;
}