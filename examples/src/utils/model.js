import fs from "fs";
import path from "path";
import {
    GanSerialization,
    GenerativeAdversarialModel,
    ModelSerialization
} from "mind-net.js";

import * as Image from "./image.js";
import * as CommonUtils from "./common.js";

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
 * @typedef {Object} SavingSampleOptions
 * @property {number} count - Sample count (default: 20)
 * @property {number} channel - Channel count (default: 1)
 * @property {number} border - Border size (default: 2)
 * @property {number} scale - Scaling value (default: 1)
 * @property {string} prefix - FileName prefix (default: "sample")
 * @property {boolean} time - Include time to fileName (default: true)
 */

/**
 * @type {SavingSampleOptions}
 */
const SavingSampleOptionsDefaults = {
    count: 20,
    channel: 1,
    border: 2,
    scale: 1,
    time: true,
    prefix: "sample"
};

/**
 * @param {string|number} key
 * @param {string} outPath
 * @param {number} imageSize
 * @param {(x: number, y: number) => number[]} dataFn
 * @param {Partial<SavingSampleOptions>} options
 *
 * @return {Promise<void>}
 */
export async function saveModelsSamples(key, outPath, imageSize, dataFn, options = {}) {
    const opts = Object.assign({}, SavingSampleOptionsDefaults, options);

    const fileName = [
        opts.prefix,
        typeof key === "number" ? key.toString().padStart(6, "0") : key,
        opts.time ? new Date().toISOString() : null
    ].filter(v => v).join("_");

    await Image.saveImageGrid(dataFn, path.join(outPath, fileName + ".png"), imageSize, opts.count, opts.channel, opts.border, opts.scale);
}

/**
 * @param {string|number} key
 * @param {string} outPath
 * @param {number[][]} samples
 * @param {Partial<SavingSampleOptions>} options
 *
 * @return {Promise<void>}
 */
export async function saveGeneratedModelsSamples(key, outPath, samples, options = {}) {
    const opts = Object.assign({}, SavingSampleOptionsDefaults, options);
    if (samples.length < opts.count ** 2) throw new Error("Not enough data");

    const imageSize = Math.sqrt(samples[0].length / opts.channel);
    if (imageSize % 1 !== 0) throw new Error("Bad sample size");

    return saveModelsSamples(key, outPath, imageSize,
        (x, y) => samples[x + y * opts.count],
        opts
    );
}