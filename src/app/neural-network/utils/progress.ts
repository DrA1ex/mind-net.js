import {FetchDataAsyncReader, FileAsyncReader, ObservableStreamLoader, ProgressFn} from "./fetch";
import * as CommonUtils from "./common";
import {ProgressUtils} from "../neural-network";

export enum Color {
    red = "\u001B[31m",
    green = "\u001B[32m",
    yellow = "\u001B[33m",
    blue = "\u001B[34m",
    magenta = "\u001B[35m",
    cyan = "\u001B[36m",
    lightgray = "\u001B[37m",
    default = "\u001B[39m",
    white = "\u001B[97m",
    black = "\u001B[30m",
    reset = "\u001B[0m"
}

export enum BackgroundColor {
    black = "\u001B[40m",
    red = "\u001B[41m",
    green = "\u001B[42m",
    yellow = "\u001B[43m",
    blue = "\u001B[44m",
    magenta = "\u001B[45m",
    cyan = "\u001B[46m",
    lightgray = "\u001B[47m",
    default = "\u001B[49m"
}

export enum SpeedCalculationKind {
    timePerIteration,
    iterationsPerSecond,
    auto
}

export enum ValueLimit {
    exclusive = "exclusive",
    inclusive = "inclusive",
}

export const Converters = {
    Bytes: CommonUtils.formatByteSize,
    TimeSpan: CommonUtils.formatTimeSpan,
    Metric: (value: number) => CommonUtils.formatUnit(value, "ops"),
    None: (value: number) => value.toString()
}

export type ProgressOptions = {
    width: number,
    color: Color,
    speedConverter: (value: number) => string,
    valueConverter: (value: number) => string,
    background: BackgroundColor,
    speed: SpeedCalculationKind,
    update: boolean,
    limit: ValueLimit,
    progressThrottle: number,
}

const ProgressOptionsDefaults: ProgressOptions = {
    width: 40,
    color: Color.cyan,
    background: BackgroundColor.default,
    speed: SpeedCalculationKind.auto,
    update: false,
    speedConverter: Converters.Metric,
    valueConverter: Converters.None,
    limit: ValueLimit.exclusive,
    progressThrottle: 1000,
}

const FetchProgressOptionsDefaults: Partial<ProgressOptions> = {
    color: Color.green,
    update: (typeof process !== "undefined"),
    valueConverter: Converters.Bytes,
    speedConverter: Converters.Bytes,
    limit: ValueLimit.inclusive,
    progressThrottle: (typeof process !== "undefined") ? 50 : 500,
}

export function* progress(total: number, options: Partial<ProgressOptions> = {}) {
    function* _iter(): Generator<[number, number]> {
        for (let i = 0; i < total; i++) {
            yield [i, total];
        }
    }

    yield* progressGenerator(_iter(), options);
}

export function* progressIterable(
    iterable: Iterable<any>, total?: number, options: Partial<ProgressOptions> = {}
) {
    const progressFn = progressCallback(options);

    let iteration = 0
    for (const iterableElement of iterable) {
        progressFn(iteration++, total ?? 0);
        yield iterableElement;
    }
}

export function* progressGenerator(
    iterable: Generator<[iteration: number, total: number]>, options: Partial<ProgressOptions> = {}
) {
    const progressFn = progressCallback(options);

    for (const [iteration, total] of iterable) {
        progressFn(iteration, total);
        yield iteration;
    }
}

export function progressCallback(options: Partial<ProgressOptions> = {}): ProgressFn {
    const opts: ProgressOptions = {...ProgressOptionsDefaults, ...options};
    const startTime = performance.now();
    let firstCall = true;

    const callback = (iteration: number, total?: number) => {
        iteration = Math.max(0, iteration ?? 0);
        total = Math.max(iteration, total ?? 0);

        const elapsedTime = (performance.now() - startTime) / 1000;
        const isFirstIter = iteration === 0;
        const progress = iteration / total;

        const speed = !isFirstIter ? elapsedTime / iteration : 0
        const estimateTime = !isFirstIter ? speed * total : 0

        const displayIterationValue = opts.limit === ValueLimit.exclusive ? iteration + 1 : iteration;
        const iterationsStr = opts.valueConverter(displayIterationValue);
        const totalStr = opts.valueConverter(total);

        let speedMethod = opts.speed == SpeedCalculationKind.auto
            ? (speed >= 1 ? SpeedCalculationKind.timePerIteration : SpeedCalculationKind.iterationsPerSecond)
            : opts.speed;

        let speedStr;
        if (speedMethod === SpeedCalculationKind.iterationsPerSecond) {
            speedStr = speed !== 0 ? `${opts.speedConverter(1 / speed)}/s` : "n/a";
        } else {
            speedStr = speed !== 0 ? `${Converters.TimeSpan(speed * 1000)}/it` : "n/a";
        }

        const output = opts.color + opts.background
            + `${Math.floor(progress * 100).toString().padStart(3, " ")}%|`
            + progressBar(progress, opts.width)
            + `| ${iterationsStr}/${totalStr} `
            + `[${formatTime(elapsedTime)}<${formatTime(estimateTime)}, `
            + `${speedStr}]`
            + Color.reset;


        const shouldOverrideLine = !firstCall && options.update;
        if (typeof process !== "undefined" && typeof process.stdout !== "undefined") {
            if (shouldOverrideLine) {
                process.stdout.write("\u001B[F");
                process.stdout.write("\u001B[2K");
            }
        } else {
            if (shouldOverrideLine) console.clear();
        }

        console.log(output);
        firstCall = false;
    }

    if (opts.progressThrottle > 0) {
        return throttle(callback, opts.limit, opts.progressThrottle);
    }

    return callback;
}

export type ProgressBatchCtrl = {
    readonly total: number;
    readonly batchSize: number;
    readonly progressFn: ProgressFn;

    currentOffset: number;

    addBatch(): void;
    add(count: number): void;
    reset(): void;
}

export function progressBatchCallback(
    batchSize: number, batchCount: number, options: Partial<ProgressOptions> = {}
): ProgressBatchCtrl {
    const progressFn = progressCallback(options);

    const ctrl: ProgressBatchCtrl = {
        currentOffset: 0,
        total: batchSize * batchCount,
        batchSize,
        progressFn: (current) => progressFn(ctrl.currentOffset + current, ctrl.total),

        addBatch: () => ctrl.currentOffset += batchSize,
        add: (count: number) => ctrl.currentOffset += count,
        reset: () => ctrl.currentOffset = 0,
    }

    return ctrl;
}

export async function fetchProgress(url: string, options: Partial<ProgressOptions> = {}): Promise<Uint8Array> {
    const fetchResponse = await fetch(url);

    const opts: Partial<ProgressOptions> = {...FetchProgressOptionsDefaults, ...options}
    const progressFn = progressCallback(opts);

    const reader = new FetchDataAsyncReader(fetchResponse);
    const loader = new ObservableStreamLoader(reader, progressFn);

    const buffer = await loader.loadChunked();
    return buffer.toTypedArray(Uint8Array);
}

export async function fileProgress(file: File, options: Partial<ProgressOptions> = {}): Promise<Uint8Array> {
    const opts: Partial<ProgressOptions> = {...FetchProgressOptionsDefaults, ...options}
    const progressFn = progressCallback(opts);

    const reader = new FileAsyncReader(file);
    const loader = new ObservableStreamLoader(reader, progressFn);

    const buffer = await loader.loadChunked();
    return buffer.toTypedArray(Uint8Array);
}

function formatTime(seconds: number) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    seconds = Math.floor(seconds) % 60;

    const times = [hours > 0 ? hours : undefined, minutes, seconds];
    return times.filter(t => t !== undefined)
        .map(t => t!.toString().padStart(2, '0'))
        .join(':');
}

function progressBar(progress: number, width: number) {
    const filledChar = '█';
    const partialChars = ['▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    const emptyChar = ' ';

    const progressWidth = Math.max(0, Math.min(1, progress)) * width;
    const filledCount = Math.floor(progressWidth);
    const partialIndex = Math.floor(progressWidth % 1 * partialChars.length);

    let progressBar = filledChar.repeat(filledCount);

    let partialWidth = 0;
    if (partialIndex > 0) {
        progressBar += partialChars[partialIndex - 1];
        partialWidth = 1;
    }

    progressBar += emptyChar.repeat(width - filledCount - partialWidth);

    return progressBar;
}

export function throttle(fn: ProgressFn, limit: ValueLimit, delay: number) {
    let timerId: number | null = null;
    let timerSetAt: number;
    let lastArgs: [number, number] | null = null;

    function _throttled(iteration: number, total: number) {
        const isFinished = limit === ValueLimit.inclusive ? iteration >= total : iteration >= total - 1;
        if (isFinished) {
            // If all iterations have been processed, clear the timer and invoke the original function.
            if (timerId !== null) clearTimeout(timerId);
            fn(iteration, total);
            return;
        }

        if (timerId !== null) {
            // If a timer is already active, store the arguments for later execution.
            lastArgs = [iteration, total];

            // If the timer should have fired but hasn't (possibly due to blocking code),
            //    invoke the function and clear the timer
            if (performance.now() - timerSetAt >= delay) {
                clearTimeout(timerId);
                timerId = null;
                _throttled(...lastArgs)
            }

            return;
        }

        // Set the timestamp when the timer is activated.
        timerSetAt = performance.now();

        timerId = setTimeout(() => {
            timerId = null;
            // Execute the stored arguments after the delay if any.
            if (lastArgs !== null) _throttled(...lastArgs);
            lastArgs = null;
        }, delay) as any;

        // Invoke the original function immediately if there was no timer
        return fn(iteration, total);
    }

    return _throttled;
}