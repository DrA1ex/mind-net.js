import Jimp from "jimp";
import {Matrix} from "mind-net.js";

const InputCache = {};

export async function saveSnapshot(network, version, count, channel = 1, border = 2, scale = 4) {
    const inSize = network.layers[0].size;
    const outSize = Math.sqrt(network.layers[network.layers.length - 1].size / channel);

    const imageSize = outSize * count + border * (count + 1)

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
            if (!InputCache[`${x},${y}`]) InputCache[`${x},${y}`] = Matrix.random_normal_1d(inSize, -1, 1);

            const data = network.compute(InputCache[`${x},${y}`]);

            const xOffset = border * (x + 1) + x * outSize;
            const yOffset = border * (y + 1) + y * outSize;
            for (let index = 0; index < data.length / channel; index++) {
                if (channel === 1) {
                    const color = Math.min(255, Math.max(0, Math.floor((data[index] + 1) / 2 * 255)));
                    img.setPixelColor(Jimp.rgbaToInt(color, color, color, 255),
                        xOffset + index % outSize,
                        yOffset + Math.trunc(index / outSize)
                    );
                } else if (channel === 3) {
                    const r = Math.min(255, Math.max(0, Math.floor((data[index * channel] + 1) / 2 * 255)));
                    const g = Math.min(255, Math.max(0, Math.floor((data[index * channel + 1] + 1) / 2 * 255)));
                    const b = Math.min(255, Math.max(0, Math.floor((data[index * channel + 2] + 1) / 2 * 255)));
                    img.setPixelColor(Jimp.rgbaToInt(r, g, b, 255),
                        xOffset + index % outSize,
                        yOffset + Math.trunc(index / outSize)
                    );
                } else {
                    throw new Error("Unsupported channel count");
                }
            }
        }
    }

    const fileName = `./out/out_${version.toString().padStart(6, "0")}.png`;
    await jimp.scale(scale, Jimp.RESIZE_NEAREST_NEIGHBOR).writeAsync(fileName);

    console.log(`File ${fileName} created`);
}