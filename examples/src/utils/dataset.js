import getPixels from "get-pixels";
import JSZip from "jszip";

import * as CommonUtils from "./common.js";

export function prepareData(pixels) {
    const channels = pixels.shape[2];
    const outChannels = 3;

    let data
    if (channels === 4) {
        const pixelCount = pixels.shape[0] * pixels.shape[1];
        data = new Array(pixelCount * outChannels);
        for (let i = 0; i < pixelCount; i++) {
            data[i * outChannels] = pixels.data[i * channels];
            data[i * outChannels + 1] = pixels.data[i * channels + 1];
            data[i * outChannels + 2] = pixels.data[i * channels + 2];
        }
    } else if (channels === 3) {
        data = Array.from(pixels.data);
    } else {
        throw new Error("Image must be in rgb or rgba format");
    }

    return data.map(value => (value / 127.5) - 1);
}

export async function loadDataset(zipFile, progressFn = undefined) {
    const result = [];
    const zip = await JSZip.loadAsync(zipFile);

    const zipFiles = Object.values(zip.files);
    let loaded = 0;
    for (const file of zipFiles) {
        if (file.dir) continue;

        const data = await file.async("arraybuffer");
        const pixels = await CommonUtils.promisify(getPixels, Buffer.from(data), CommonUtils.getMimeType(file.name));

        if (progressFn) progressFn(++loaded, zipFiles.length);
        result.push(prepareData(pixels));
    }

    return result;
}