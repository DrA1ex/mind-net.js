import {expose} from "threads/worker";
import {Observable} from "threads/observable";

import {IModel} from "../../neural-network/engine/base";
import {FetchDataAsyncReader, IAsyncReader, StreamAsyncReader} from "../../neural-network/utils/fetch";
import {
    ChainModel,
    UniversalModelSerializer,
    BinarySerializer,
    FileAsyncReader,
    ObservableStreamLoader,
    ImageUtils,
    ColorUtils
} from "../../neural-network/neural-network";
import {Progress, ModelSourceData, ModelSourceType, ModelParams} from "./demo3.worker.base";

const progress = new Progress();
let model: IModel;

const WorkerImpl = {
    async loadModel(modelLoadEntries: ModelSourceData[]): Promise<ModelParams> {
        progress.total = modelLoadEntries.reduce((p, c) => p + c.size, 0);

        try {
            const chain = new ChainModel();
            for (const loadEntry of modelLoadEntries) {
                const reader = await _getModerDataReader(loadEntry);
                const loader = new ObservableStreamLoader(reader, progress.progressFn.bind(progress));

                const data = (await loader.loadChunked()).toTypedArray(Int8Array).buffer;
                if (loadEntry.name.endsWith(".json")) {
                    const config = JSON.parse(new TextDecoder().decode(data));
                    const model = UniversalModelSerializer.load(config, true);
                    chain.addModel(model);
                } else {
                    const model = BinarySerializer.load(data)
                    chain.addModel(model);
                }

                progress.offset += loadEntry.size;
            }

            model = chain;
            model.compile();

            const sizes = model.layers.map(l => {
                const size = Math.sqrt(l.size);
                if (Number.isInteger(size)) {
                    return `${size}²`;
                }

                return l.size.toString();
            });

            return {
                inputSize: model.inputSize,
                outputSize: model.outputSize,
                description: `${sizes.join(" -> ")}`
            };
        } finally {
            progress.reset();
        }
    },

    async compute(data: Uint8ClampedArray) {
        if (!model) throw new Error("Model is not loaded");

        const input = ColorUtils.transformChannelCount(Array.from(data), 4, 3);
        ColorUtils.transformColorSpace(ColorUtils.rgbToTanh, input, 3, input);

        const output3 = ImageUtils.processMultiChannelData(model, input, 3);
        ColorUtils.transformColorSpace(ColorUtils.tanhToRgb, output3, 3, output3);

        const output4 = ColorUtils.transformChannelCount(output3, 3, 4);
        const outData = new Uint8ClampedArray(output4);
        const outSize = Math.sqrt(model.outputSize);

        return {buffer: outData.buffer, size: outSize};
    },

    progress() {
        return Observable.from(progress.subject);
    }
}

expose(WorkerImpl);

export type WorkerT = typeof WorkerImpl;

async function _getModerDataReader(loadEntry: ModelSourceData): Promise<IAsyncReader> {
    if (loadEntry.source === ModelSourceType.file) {
        return new FileAsyncReader(loadEntry.data as File);
    } else if (loadEntry.source === ModelSourceType.remote) {
        const response = await fetch(loadEntry.data as string);
        return new FetchDataAsyncReader(response);
    } else if (loadEntry.source === ModelSourceType.stream) {
        return new StreamAsyncReader(loadEntry.data as ReadableStream);
    } else if (loadEntry.source === ModelSourceType.buffer) {
        return {
            size: loadEntry.size,
            async* [Symbol.asyncIterator]() {
                yield new Uint8Array(loadEntry.data as ArrayBuffer, 0, loadEntry.size);
            }
        } as IAsyncReader
    }

    throw new Error(`Unsupported source: ${loadEntry.source}`);
}