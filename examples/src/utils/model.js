import fs from "fs";
import path from "path";
import {JsonStreamStringify} from 'json-stream-stringify';

import {
    GanSerialization,
    GenerativeAdversarialModel,
    UniversalModelSerializer,
    BinarySerializer,
    TensorType
} from "mind-net.js";

import * as Image from "./image.js";
import * as CommonUtils from "./common.js";

export async function saveModel(model, fileName) {
    let dump;
    if (model instanceof GenerativeAdversarialModel) {
        dump = GanSerialization.save(model);
    } else {
        dump = UniversalModelSerializer.save(model);
    }

    const jsonStream = new JsonStreamStringify(dump);
    const fileStream = fs.createWriteStream(fileName);
    jsonStream.pipe(fileStream);

    return new Promise((resolve, reject) => {
        fileStream.on("close", () => {
            console.log(`Saved ${fileName}`);
            resolve();
        });

        fileStream.on("error", reject);
    });
}

export async function saveModelBinary(model, fileName) {
    const dump = BinarySerializer.save(model, TensorType.F32);
    return await CommonUtils.promisify(fs.writeFile, fileName, new Uint8Array(dump));
}

/**
 * @param {{[key: string]: IModel|GenerativeAdversarialModel}} models
 * @param {string} outPath
 * @param {boolean} [binary=false]
 * @return {Promise<void>}
 */
export async function saveModels(models, outPath, binary = false) {
    const time = new Date().toISOString();
    const count = Object.keys(models).length;
    console.log(`Saving ${count} models...`);

    if (!fs.existsSync(outPath)) {
        await CommonUtils.promisify(fs.mkdir, outPath, {recursive: true})
    }

    for (const [key, model] of Object.entries(models)) {
        const epoch = model.epoch ?? model.ganChain?.epoch ?? 0;

        const fileName = path.join(outPath, `${key}_${time}_${epoch}`);
        if (binary) {
            await saveModelBinary(model, fileName + ".bin");
        } else {
            await saveModel(model, fileName + ".json");
        }
    }
}

/**
 * @typedef {Object} SavingSampleOptions
 * @property {number} count - Sample count (default: 20)
 * @property {number} channel - Channel count (default: 1)
 * @property {number} border - Border size (default: 2)
 * @property {number} scale - Scaling value (default: 1)
 * @property {string} prefix - FileName prefix (default: "sample")
 * @property {string} suffix - FileName suffix (default: "")
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
    prefix: "sample",
    suffix: "",
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
        opts.suffix,
        opts.time ? new Date().toISOString() : null
    ].filter(v => v).join("_");

    await Image.saveImageGrid(dataFn, path.join(outPath, fileName + ".png"), imageSize, opts.count, opts.channel, opts.border, opts.scale);
}

/**
 * @param {string|number} key
 * @param {string} outPath
 * @param {Matrix1D[]} samples
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