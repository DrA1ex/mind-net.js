import fs from "fs";
import path from "path";
import {GanSerialization, GenerativeAdversarialModel, Matrix, ModelSerialization} from "mind-net.js";
import * as ImageUtils from "./image.js";
import * as CommonUtils from "./common.js";

export function processMultiChannelData(network, src, channels = 3, dst = null) {
    const channelSize = src.length / channels;
    if (channelSize % 1 !== 0) throw new Error(`Invalid input data size`);

    const outSize = network.layers[network.layers.length - 1].size;
    const result = dst ?? new Array(outSize * channels);

    const channelData = new Array(channelSize);
    for (let c = 0; c < 3; c++) {
        ImageUtils.getChannel(src, c, 3, channelData);
        const processedChannel = network.compute(channelData);
        ImageUtils.setChannel(result, processedChannel, c, channels);
    }

    return result;
}

export async function saveModel(model, fileName) {
    let dump;
    if (model instanceof GenerativeAdversarialModel) {
        dump = GanSerialization.save(model);
    } else {
        dump = ModelSerialization.save(model);
    }

    await CommonUtils.promisify(fs.writeFile, fileName, JSON.stringify(dump));
    console.log(`Saved ${fileName}`);
}

/**
 * @param {{[key: string]: IModel|GenerativeAdversarialModel}} models
 * @param {string} outPath
 * @return {Promise<void>}
 */
export async function saveModels(models, outPath) {
    const time = new Date().toISOString();
    const count = Object.keys(models).length;
    console.log(`Saving ${count} models...`);

    if (!fs.existsSync(outPath)) {
        await CommonUtils.promisify(fs.mkdir, outPath, {recursive: true})
    }

    for (const [key, model] of Object.entries(models)) {
        const epoch = model.epoch ?? model.ganChain?.epoch ?? 0;

        const fileName = path.join(outPath, `${key}_${time}_${epoch}.json`);
        await saveModel(model, fileName);
    }
}

/**
 * @param {string} key
 * @param {string} outPath
 * @param {number} imageSize
 * @param {(x: number, y: number) => number[]} dataFn
 * @param {number} [count=20]
 * @param {number} [channel = 1]
 * @param {number} [border = 2]
 * @param {number} [scale = 1]
 *
 * @return {Promise<void>}
 */
export async function saveModelsSamples(key, outPath, imageSize, dataFn, {
    count = 20, channel = 1, border = 2, scale = 1
} = {}) {
    const time = new Date().toISOString();
    const fileName = path.join(outPath, `sample_${key}_${time}.png`);
    await ImageUtils.saveImageGrid(dataFn, fileName, imageSize, count, channel, border, scale);
}