import * as ImageUtils from "./image.js";

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