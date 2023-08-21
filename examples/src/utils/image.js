import Jimp from "jimp";
import {Matrix} from "mind-net.js";

export const InputCache = new Map();

export async function saveSnapshot(network, version, {
    label = "out", count = 10, channel = 1, border = 2, scale = 4
}) {
    const inSize = network.layers[0].size;
    const outSize = Math.sqrt(network.layers[network.layers.length - 1].size / channel);

    if (!InputCache.has(network)) InputCache.set(network, {});
    const networkCache = InputCache.get(network);

    function _data(x, y) {
        if (!networkCache[`${x},${y}`]) networkCache[`${x},${y}`] = Matrix.random_normal_1d(inSize, -1, 1);
        return network.compute(networkCache[`${x},${y}`]);
    }

    const nameParts = [
        label, version.toString().padStart(6, "0")
    ].filter(v => v && v.length > 0);

    const fileName = `./out/${nameParts.join("_")}.png`;
    await saveImageGrid(_data, fileName, outSize, count, channel, border, scale);
}

export async function saveImageGrid(dataFn, path, size, count, channel = 1, border = 2, scale = 4) {
    const imageSize = size * count + border * (count + 1)

    const {jimp, img} = await new Promise((resolve, reject) => {
        const jimp = new Jimp(imageSize, imageSize, "#fff", (err, img) => {
            if (!err) {
                resolve({jimp, img})
            } else {
                reject(err);
            }
        });
    });

    for (let x = 0; x < count; x++) {
        for (let y = 0; y < count; y++) {
            const data = dataFn(x, y);

            const xOffset = border * (x + 1) + x * size;
            const yOffset = border * (y + 1) + y * size;
            for (let index = 0; index < data.length / channel; index++) {
                const hexColor = getColor(data, index * channel, channel);
                img.setPixelColor(
                    hexColor,
                    xOffset + index % size,
                    yOffset + Math.trunc(index / size)
                );
            }
        }
    }

    await jimp.scale(scale, Jimp.RESIZE_NEAREST_NEIGHBOR).writeAsync(path);
    console.log(`File ${path} created`);
}

export function grayscaleDataset(data, channels = 3) {
    if (channels !== 3 && channels !== 4) throw new Error("Unsupported channel count");

    const result = new Array(data.length);
    for (let i = 0; i < result.length; i++) {
        const gsSize = data[i].length / channels;

        result[i] = new Array(gsSize);
        for (let j = 0; j < gsSize; j++) {
            // Calculate grayscale value using the luminosity method
            result[i][j] = 0.2989 * data[i][j * channels]
                + 0.587 * data[i][j * channels + 1]
                + 0.114 * data[i][j * channels + 2];
        }
    }

    return result;
}

export function getChannel(data, channel, channelCount, dst = null) {
    const result = dst ?? new Array(data.length / channelCount);
    for (let k = 0; k < result.length; k++) {
        result[k] = data[k * channelCount + channel];
    }

    return result;
}

export function setChannel(out, data, channel, channelCount) {
    for (let k = 0; k < data.length; k++) {
        out[k * channelCount + channel] = data[k];
    }
}

function _convertPixel(value) {
    return Math.min(255, Math.max(0, Math.floor((value + 1) / 2 * 255)));
}

function getColor(data, index, channel) {
    if (channel === 1) {
        const color = _convertPixel(data[index]);
        return Jimp.rgbaToInt(color, color, color, 255);
    }

    if (channel === 3 || channel === 4) {
        const r = _convertPixel(data[index]);
        const g = _convertPixel(data[index + 1]);
        const b = _convertPixel(data[index + 2]);
        const a = channel === 4 ? _convertPixel(data[index + 3]) : 255;

        return Jimp.rgbaToInt(r, g, b, a);
    }

    throw new Error("Unsupported channel count");
}