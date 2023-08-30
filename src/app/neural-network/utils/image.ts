import {ParallelModelWrapper, Matrix} from "../neural-network";
import {Matrix1D, Matrix2D} from "../engine/matrix";
import {IModel} from "../engine/base";

export function grayscaleDataset(data: Matrix2D, channels = 3) {
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

export function getChannel(data: Matrix1D, channel: number, channelCount: number, dst = null) {
    const result = dst ?? new Array(data.length / channelCount);
    for (let k = 0; k < result.length; k++) {
        result[k] = data[k * channelCount + channel];
    }

    return result;
}

export function setChannel(out: Matrix1D, data: Matrix1D, channel: number, channelCount: number) {
    for (let k = 0; k < data.length; k++) {
        out[k * channelCount + channel] = data[k];
    }
}

export function splitChunks(data: Matrix1D, imageSize: number, cropSize: number) {
    if (data.length !== imageSize * imageSize) throw new Error("Invalid image size")

    const dimFactor = imageSize / cropSize;
    if (dimFactor % 1 !== 0) throw new Error("Sizes must be multiples of each other");

    const chunkCount = dimFactor * dimFactor;
    const result = new Array(chunkCount);

    for (let y = 0; y < dimFactor; y++) {
        for (let x = 0; x < dimFactor; x++) {
            result[y * dimFactor + x] = crop(data, x * cropSize, y * cropSize, imageSize, cropSize);
        }
    }

    return result;
}

export function crop(data: Matrix1D, x: number, y: number, imageSize: number, cropSize: number) {
    if (x + cropSize > imageSize) throw new Error("Invalid x offset");
    if (y + cropSize > imageSize) throw new Error("Invalid y offset");

    const chunk = new Array(cropSize * cropSize);
    for (let i = 0; i < cropSize; i++) {
        const yOffset = (y + i) * imageSize;
        for (let j = 0; j < cropSize; j++) {
            chunk[i * cropSize + j] = data[yOffset + x + j];
        }
    }

    return chunk;
}

export function joinChunks(chunks: Matrix2D) {
    const chunkCount = chunks.length
    const chunkLength = chunks[0].length;
    const totalSize = chunkLength * chunkCount;

    const imageSize = Math.sqrt(totalSize);
    const cropSize = Math.sqrt(chunkLength);

    if (!Number.isFinite(cropSize) || cropSize % 1 !== 0) throw new Error("Invalid chunk size");
    if (!Number.isFinite(imageSize) || imageSize % 1 !== 0) throw new Error("Invalid chunk count");

    const dimFactor = imageSize / cropSize;
    if (dimFactor % 1 !== 0) throw new Error("Sizes must be multiples of each other");

    const result = new Array(totalSize);
    for (let y = 0; y < dimFactor; y++) {
        for (let x = 0; x < dimFactor; x++) {
            setCroppedImage(chunks[y * dimFactor + x], result,
                x * cropSize, y * cropSize, imageSize, cropSize);
        }
    }

    return result;
}

export function setCroppedImage(chunk: Matrix1D, dst: Matrix1D, x: number, y: number, imageSize: number, cropSize: number) {
    if (x + cropSize > imageSize) throw new Error("Invalid x offset");
    if (y + cropSize > imageSize) throw new Error("Invalid y offset");

    for (let i = 0; i < cropSize; i++) {
        const yOffset = (y + i) * imageSize;
        for (let j = 0; j < cropSize; j++) {
            dst[yOffset + x + j] = chunk[i * cropSize + j];
        }
    }
}

export async function processMultiChunkDataParallel<T extends IModel>(
    pModel: ParallelModelWrapper<T>, inputs: Matrix2D, channels = 1
) {
    const imageSize = Math.sqrt(inputs[0].length / channels);
    const cropSize = Math.sqrt(pModel.model.inputSize);

    if (imageSize % 1 !== 0) throw new Error("Wrong input image size");
    if (cropSize % 1 !== 0) throw new Error("Wrong input model size");
    if (!(pModel instanceof ParallelModelWrapper)) throw new Error("Model should be ParallelModelWrapper");

    const chunks = inputs.map(input =>
        Matrix.fill(
            c => splitChunks(getChannel(input, c, channels), imageSize, cropSize),
            channels
        ).flat()
    ).flat();

    const batchSize = Math.min(Math.ceil(chunks.length / pModel.parallelism), 128);
    const outChunks = await pModel.compute(chunks, {batchSize});
    const chunksPerInput = chunks.length / inputs.length;
    const chunkPerChannel = chunksPerInput / channels;

    const oSize = pModel.model.outputSize * chunkPerChannel * channels;
    const result = new Float64Array(inputs.length * oSize);

    const outputs = new Array(inputs.length);
    for (let i = 0; i < inputs.length; i++) {
        const slice = result.subarray(i * oSize, (i + 1) * oSize);
        outputs[i] = slice;

        const chunksOffset = i * chunksPerInput;
        for (let c = 0; c < channels; c++) {
            const chunkSlice = outChunks.slice(chunksOffset + c * chunkPerChannel, chunksOffset + (c + 1) * chunkPerChannel);
            const processed = joinChunks(chunkSlice);
            setChannel(slice as any, processed, c, channels);
        }
    }

    return outputs;
}

export async function processMultiChannelDataParallel<T extends IModel>(
    pModel: ParallelModelWrapper<T>, inputs: Matrix2D, channels = 3
) {
    const inputSize = inputs[0].length
    const channelSize = inputSize / channels;
    if (!Number.isFinite(channelSize) || channelSize % 1 !== 0) throw new Error("Invalid input data size");
    if (!(pModel instanceof ParallelModelWrapper)) throw new Error("Model should be ParallelModelWrapper");

    const channelsData = inputs.map(input =>
        Matrix.fill(c => getChannel(input, c, channels), channels)
    ).flat();

    const processedChannels = await pModel.compute(channelsData);

    const outSize = pModel.model.outputSize * channels;
    const result = new Float64Array(inputs.length * outSize);

    const outputs = new Array(inputs.length);
    for (let i = 0; i < inputs.length; i++) {
        const slice = result.subarray(i * outSize, (i + 1) * outSize);
        outputs[i] = slice;

        for (let c = 0; c < channels; c++) {
            const processed = processedChannels[i * channels + c];
            setChannel(slice as any, processed, c, channels);
        }
    }

    return outputs;
}