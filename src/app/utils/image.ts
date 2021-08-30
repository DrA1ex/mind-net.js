// @ts-ignore
import {PNG} from 'pngjs/browser';

export function getTrainingDataFromImage(buffer: ArrayBuffer): Promise<number[]> {
    return new Promise((resolve, reject) => {
        try {
            new PNG().parse(buffer, (error: any, data: any) => {
                if (error) {
                    return reject(error);
                }

                const size = data.height * data.width;
                const bytesPerPixel = data.data.length / size;

                const result = new Array(size);

                for (let i = 0; i < size; i++) {
                    // get only Red channel value
                    result[i] = data.data[i * bytesPerPixel] / 255;
                }

                resolve(result);
            });
        } catch (err) {
            reject(err);
        }
    });
}