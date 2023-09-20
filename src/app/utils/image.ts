// @ts-ignore
import {PNG} from 'pngjs/browser';
import {ImageUtils} from "../neural-network/neural-network";
import {Matrix1D} from "../neural-network/engine/matrix";

export function getTrainingDataFromImage(buffer: ArrayBuffer): Promise<Matrix1D> {
    return new Promise((resolve, reject) => {
        try {
            new PNG().parse(buffer, (error: any, data: any) => {
                if (error) {
                    return reject(error);
                }

                const size = data.height * data.width;
                const bytesPerPixel = data.data.length / size;

                const gsData = ImageUtils.grayscaleDataset([data.data], bytesPerPixel)[0];
                for (let i = 0; i < gsData.length; i++) {
                    gsData[i] = gsData[i] / 255;
                }

                resolve(gsData);
            });
        } catch (err) {
            reject(err);
        }
    });
}