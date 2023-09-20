import {expose} from "threads/worker";
import {Observable, Subject} from "threads/observable";

import {IModel} from "../../neural-network/engine/base";
import {ProgressFn} from "../../neural-network/utils/fetch";
import {
    ChainModel,
    UniversalModelSerializer,
    BinarySerializer,
    FileAsyncReader,
    ObservableStreamLoader,
    ImageUtils,
    ColorUtils,
    ProgressUtils
} from "../../neural-network/neural-network";

class Progress {
    private _current = 0;
    private _total = 0;

    private readonly _throttledSendProgress: ProgressFn;

    readonly subject = new Subject();
    offset = 0;

    get current() {return this._current;}
    set current(value: number) {
        this._current = value;
        this.refresh();
    }

    get total() {return this._total;}
    set total(value: number) {
        this._total = value;
        this.refresh();
    }

    constructor(delay = 100) {
        this._throttledSendProgress = ProgressUtils.throttle(
            (current, total) => this.subject.next({current, total}),
            ProgressUtils.ValueLimit.inclusive,
            delay
        );
    }

    reset() {
        this.offset = 0;
        this._current = 0;
        this._total = 0;

        this.refresh();
    }

    refresh() {
        this._throttledSendProgress(this.offset + this.current, this.total);
    }

    progressFn(current: number, _: number) {
        this.current = current;
    }
}

const progress = new Progress();
let model: IModel;

export type ModelParams = {
    inputSize: number
    outputSize: number
    description: string
}

const WorkerImpl = {
    async loadModel(files: File[]): Promise<ModelParams> {
        progress.total = files.reduce((p, c) => p + c.size, 0);

        try {
            const chain = new ChainModel();
            for (const file of files) {
                const reader = new FileAsyncReader(file);
                const loader = new ObservableStreamLoader(reader, progress.progressFn.bind(progress));

                const data = await loader.load();
                if (file.name.endsWith(".json")) {
                    const config = JSON.parse(new TextDecoder().decode(data));
                    const model = UniversalModelSerializer.load(config, true);
                    chain.addModel(model);
                } else {
                    const model = BinarySerializer.load(data)
                    chain.addModel(model);
                }

                progress.offset += file.size;
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