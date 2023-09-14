import {jest} from "@jest/globals";
import * as ArrayUtils from "./utils/array";

import {ColorUtils, FetchDataAsyncReader, Matrix, ProgressUtils} from "../src/app/neural-network/neural-network";
import {ValueLimit} from "../src/app/neural-network/utils/progress";
import {ObservableStreamLoader} from "../src/app/neural-network/utils/fetch";
import {ChunkedArrayBuffer} from "../src/app/neural-network/utils/array-buffer";

jest.useFakeTimers();
jest.spyOn(global, "clearTimeout");
jest.spyOn(console, "log").mockImplementation(() => {});

function _mockFetch(data: number[][], headers: [string, string][] = []) {
    const readMock = jest.fn<any>();
    for (const chunk of data) {
        readMock.mockResolvedValueOnce({done: false, value: new Uint8Array(chunk)});
    }
    readMock.mockResolvedValueOnce({done: true});

    const mockResponse = {
        ok: true,
        body: {getReader: () => ({read: readMock})},
        headers: new Map<string, string>(headers)
    };

    const fetchMock = jest.spyOn(global, "fetch");
    fetchMock.mockResolvedValueOnce(mockResponse as any);

    return {fetchMock, readMock, mockResponse};
}

describe("throttle", () => {
    beforeEach(() => {
        // Reset the mock function and timers before each test
        jest.clearAllMocks();
        jest.clearAllTimers();
    });

    const mockFn = jest.fn();

    it("should invoke the callback immediately", () => {
        const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.exclusive, 500);

        throttledFn(1, 10);

        // Verify that the callback is called immediately
        expect(mockFn).toHaveBeenCalledTimes(1);
        expect(mockFn).toHaveBeenCalledWith(1, 10);

        mockFn.mockReset();
        jest.advanceTimersByTime(1000);

        // Verify that the timer is not used in this case
        expect(mockFn).not.toHaveBeenCalled();
    });

    it("should throttle subsequent invocations", () => {
        const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.exclusive, 500);

        throttledFn(1, 10);
        throttledFn(2, 10);
        throttledFn(3, 10);

        // Verify that the callback is called only once immediately
        expect(mockFn).toHaveBeenCalledTimes(1);
        expect(mockFn).toHaveBeenCalledWith(1, 10);

        // Fast-forward time by 500ms
        jest.advanceTimersByTime(500);

        // Verify that the callback is called again with the last arguments
        expect(mockFn).toHaveBeenCalledTimes(2);
        expect(mockFn).toHaveBeenCalledWith(3, 10);

        mockFn.mockReset();
        jest.advanceTimersByTime(1000);

        // Verify that the timer is not used in this case
        expect(mockFn).not.toHaveBeenCalled();
    });

    it("should clear the timer on completion", () => {
        const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.exclusive, 500);

        throttledFn(1, 10);

        // Verify that the callback is called immediately
        expect(mockFn).toHaveBeenCalledTimes(1);
        expect(mockFn).toHaveBeenCalledWith(1, 10);

        jest.advanceTimersByTime(100);
        throttledFn(9, 10);

        // Verify that the timer is cleared after completion
        expect(clearTimeout).toHaveBeenCalled();
    });

    it("should clear the timer when timer was missed", () => {
        const performanceMock = jest.spyOn(performance, "now");
        performanceMock.mockReturnValue(0);

        const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.exclusive, 500);

        throttledFn(1, 10);
        expect(mockFn).toHaveBeenCalledTimes(1);

        // Simulate missed time
        performanceMock.mockReset().mockReturnValue(1000);

        throttledFn(5, 10);
        // Should be force called by missed timer condition
        expect(mockFn).toHaveBeenCalledTimes(2);
        expect(mockFn).toHaveBeenCalledWith(5, 10);
        // Verify that the timer is cleared after completion
        expect(clearTimeout).toHaveBeenCalled();

        jest.advanceTimersByTime(1000);

        // Verify that timer will not fire again
        expect(mockFn).toHaveBeenCalledTimes(2);
    });

    describe("inclusive limit", () => {
        it("should call the original function if all iterations have been processed", () => {
            const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.inclusive, 100);
            throttledFn(5, 5);

            expect(mockFn).toHaveBeenCalledTimes(1);
            expect(mockFn).toHaveBeenCalledWith(5, 5);
        });


        it("should delay the function when limit is not reached", async () => {
            const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.inclusive, 100);
            throttledFn(2, 5);
            expect(mockFn).toHaveBeenCalled();

            mockFn.mockReset();
            throttledFn(4, 5);
            expect(mockFn).not.toHaveBeenCalled();
        });
    });

    describe("exclusive limit", () => {
        it("should call the original function if all iterations have been processed", () => {
            const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.exclusive, 100);
            throttledFn(5, 6);
            expect(mockFn).toHaveBeenCalledTimes(1);
            expect(mockFn).toHaveBeenCalledWith(5, 6);
        });


        it("should delay the function when limit is not reached", async () => {
            const throttledFn = ProgressUtils.throttle(mockFn, ValueLimit.exclusive, 100);
            throttledFn(2, 5);
            expect(mockFn).toHaveBeenCalled();

            mockFn.mockReset();
            throttledFn(3, 5);
            expect(mockFn).not.toHaveBeenCalled();
        });
    })
});

describe("progress", () => {
    it("generates the correct progress sequence", () => {
        const total = 5;
        const iterator = ProgressUtils.progress(total);
        const result = Array.from(iterator);

        expect(result).toEqual([0, 1, 2, 3, 4]);
    });

    it.each([ValueLimit.inclusive, ValueLimit.exclusive])
    ("generates the correct progress sequence with options (%p)", (limit) => {
        const total = 5;
        const options = {color: ProgressUtils.Color.green, limit};
        const iterator = ProgressUtils.progress(total, options);
        const result = Array.from(iterator);

        expect(result).toEqual([0, 1, 2, 3, 4]);
    });

    it("generates the correct progress sequence from an iterable", () => {
        const iterable = [[0, 3], [1, 3], [2, 3]];
        const iterator = ProgressUtils.progressIterable(iterable);
        const result = Array.from(iterator);

        expect(result).toEqual(iterable);
    });

    it.each([ValueLimit.inclusive, ValueLimit.exclusive])
    ("generates the correct progress sequence from an iterable with options (%p)", (limit) => {
        const iterable = [[0, 3], [1, 3], [2, 3]];
        const options = {color: ProgressUtils.Color.green, limit};
        const iterator = ProgressUtils.progressIterable(iterable, iterable.length, options);
        const result = Array.from(iterator);

        expect(result).toEqual(iterable);
    });
});

describe("fetchProgress", () => {
    it("should fetch data and return Uint8Array", async () => {
        const {fetchMock} = _mockFetch([[1, 2, 3]]);
        const result = await ProgressUtils.fetchProgress("https://example.com");

        expect(fetchMock).toHaveBeenCalledWith("https://example.com");
        expect(result).toEqual(new Uint8Array([1, 2, 3]));
    });

    it("should fetch data with Content-Length", async () => {
        const {fetchMock} = _mockFetch([[1, 2, 3]], [["Content-Length", "3"]]);
        const result = await ProgressUtils.fetchProgress("https://example.com");

        expect(fetchMock).toHaveBeenCalledWith("https://example.com");
        expect(result).toEqual(new Uint8Array([1, 2, 3]));
    });
});

describe("FetchDataAsyncReader", () => {
    it("should read chunks", async () => {
        const testChunks = [[1, 2, 3], [4, 5, 6], [7], [8, 9], [10]];
        const {mockResponse, readMock} = _mockFetch(testChunks);

        const reader = new FetchDataAsyncReader(mockResponse as any);
        const chunks = [];
        for await (const chunk of reader) {
            chunks.push(chunk);
        }

        const expectedChunks = testChunks.map(c => new Uint8Array(c));
        expect(chunks).toEqual(expectedChunks);
        expect(readMock).toHaveBeenCalledTimes(testChunks.length + 1);
    });

    it("should read chunks with Content-Length", async () => {
        const {mockResponse, readMock} = _mockFetch([[1, 2, 3], [4, 5, 6]], [["Content-Length", "6"]]);

        const reader = new FetchDataAsyncReader(mockResponse as any);
        expect(reader.size).toBe(6);

        const chunks = [];
        for await (const chunk of reader) {
            chunks.push(chunk);
        }

        expect(chunks).toEqual([new Uint8Array([1, 2, 3]), new Uint8Array([4, 5, 6])]);
        expect(readMock).toHaveBeenCalledTimes(3);
    });
});

describe('ObservableStreamLoader', () => {
    describe('load', () => {
        it('should load the data from the stream and return a buffer', async () => {
            const streamSize = 1024;
            const fakeData = new Uint8Array(streamSize);
            const asyncReader = {
                size: streamSize,
                [Symbol.asyncIterator]: async function* () {yield fakeData;}
            }

            const progressFn = jest.fn();

            const loader = new ObservableStreamLoader(asyncReader, progressFn);
            const result = await loader.load();

            expect(progressFn).toHaveBeenCalledWith(0, streamSize);
            expect(progressFn).toHaveBeenCalledWith(streamSize, streamSize);
            expect(result).toBeInstanceOf(ArrayBuffer);
            expect(new Uint8Array(result)).toEqual(fakeData);
        });
    });

    describe('loadChunked', () => {
        it('should load the data from the stream in chunks and return a ChunkedArrayBuffer', async () => {
            const streamSize = 1024;
            const chunkSize = 256;
            const fakeData = new Uint8Array(Matrix.random_1d(streamSize, 0, 255));
            const asyncReader = {
                size: streamSize,
                [Symbol.asyncIterator]: async function* () {
                    for (let offset = 0; offset < streamSize; offset += chunkSize) {
                        yield fakeData.slice(offset, offset + chunkSize);
                    }
                }
            }

            const progressFn = jest.fn();

            const loader = new ObservableStreamLoader(asyncReader, progressFn);
            const result = await loader.loadChunked(chunkSize);

            expect(progressFn).toHaveBeenCalledWith(0, streamSize);
            expect(progressFn).toHaveBeenCalledWith(streamSize, streamSize);
            expect(result).toBeInstanceOf(ChunkedArrayBuffer);

            expect(result.toTypedArray(Uint8Array)).toStrictEqual(fakeData);
        });
    });
});

describe("ColorUtils", () => {
    describe("channels count transformation", () => {
        test.each([
            {from: [0, 128, 0], to: [0, 128, 0, 255]},
            {from: [123, 212, 15, 128], to: [123, 212, 15]},
            {from: [50], to: [50, 50, 50]},
            {from: [120], to: [120, 120, 120, 255]},
            {from: [20, 30, 40], to: [28.148]},
            {from: [50, 60, 90, 255], to: [60.425]}
        ])("$from.length -> $to.length", ({from, to}) => {
            const result = ColorUtils.transformChannelCount(from, from.length, to.length);
            ArrayUtils.arrayCloseTo(result, to);
        })
    });

    describe("channels count transformation invalid channels count", () => {
        test.failing.each([
            {from: 4, to: 4},
            {from: 3, to: 2},
            {from: 2, to: 3},
            {from: 1, to: 2},
            {from: 4, to: 2},
            {from: 1, to: 1},
        ])
        ("$from -> $to", ({from, to}) => {
            ColorUtils.transformChannelCount(new Array(from).fill(0), from, to);
        });
    })

    const ColorEps = 1e-2;
    const ColorRoughEps = 1;

    const LabToRgbTestData = [
        {lab: [0, 128, -128], rgb: [0, 0, 195]},
        {lab: [50, 0, 50], rgb: [141, 117, 25]},
        {lab: [80, 35, 30], rgb: [255, 172, 144]},
        {lab: [90, -50, -15], rgb: [0, 253, 253]},
        {lab: [2.5, 0, 0.5], rgb: [10, 9, 8]},
    ] as { lab: [number, number, number], rgb: [number, number, number] }[];

    describe("lab -> rgb", () => {
        test.each(LabToRgbTestData)
        ("lab: $lab, rgb: $rgb", ({lab, rgb}) => {
            const transformed = ColorUtils.labToRgb(...lab)
            ArrayUtils.arrayCloseTo(rgb, transformed, ColorRoughEps);
        })
    });

    const RgbToLabTestData = [
        {rgb: [0, 0, 195], lab: [23, 65, -88]},
        {rgb: [128, 100, 25], lab: [44, 4, 44]},
        {rgb: [255, 150, 50], lab: [72, 33, 66]},
        {rgb: [0, 240, 230], lab: [86, -49, -9]},
        {rgb: [10, 9, 8], lab: [3, 0, 0]},
        {rgb: [255, 255, 255], lab: [100, 0, 0]},
        {rgb: [0, 0, 0], lab: [0, 0, 0]}
    ] as { lab: [number, number, number], rgb: [number, number, number] }[];

    describe("rgb -> lab", () => {
        test.each(RgbToLabTestData)
        ("rgb: $rgb, lab: $lab", ({lab, rgb}) => {
            const transformed = ColorUtils.rgbToLab(...rgb)
            ArrayUtils.arrayCloseTo(lab, transformed, ColorRoughEps);
        })
    });

    const RgbTestValues = [
        {rgb: [120, 0, 50]},
        {rgb: [70, 50, 50]},
        {rgb: [40, 255, 20]},
        {rgb: [100, 0, 50]},
        {rgb: [255, 0, 255]},
        {rgb: [48.51, 21.32, 241.75]},
        {rgb: [123, 211, 122]},
        {rgb: [255, 255, 255]},
        {rgb: [0, 0, 0]},
        {rgb: [256, 255.5, 500]},
        {rgb: [-100, -1, -0.5]},
    ] as { rgb: [number, number, number] }[];

    describe("rgb <-> lab", () => {
        test.each(RgbTestValues)
        ("$rgb", ({rgb}) => {
            const lab = ColorUtils.rgbToLab(...rgb);
            const rgbFromLab = ColorUtils.labToRgb(...lab);

            ArrayUtils.arrayCloseTo(ColorUtils.clampRgb(...rgb), rgbFromLab, ColorEps);
        })
    });

    describe("rgb <-> tanh", () => {
        test.each(RgbTestValues)
        ("$rgb", ({rgb}) => {
            const tanh = ColorUtils.rgbToTanh(...rgb);
            const rgbFromTanh = ColorUtils.tanhToRgb(...tanh);

            ArrayUtils.arrayCloseTo(ColorUtils.clampRgb(...rgb), rgbFromTanh, ColorEps);
        });
    });

    const LabTestData = [
        {lab: [8.56, 28.41, 13.52]},
        {lab: [33.11, 36.27, -71.61]},
        {lab: [57.11, 69.08, 68.74]},
        {lab: [70.81, -66.64, 49.75]},
        {lab: [13.92, -13.42, 10.69]},
        {lab: [26.85, 20.00, 28.28]},
        {lab: [83.22, 5.73, 84.54]},
        {lab: [40.91, 83.17, -93.29]},
        {lab: [0, 0, 0]}
    ] as { lab: [number, number, number] }[];

    describe("lab <-> rgb", () => {
        test.each(LabTestData)
        ("$lab", ({lab}) => {
            const rgb = ColorUtils.labToRgb(...lab);
            const labFromRgb = ColorUtils.rgbToLab(...rgb);

            ArrayUtils.arrayCloseTo(ColorUtils.clampLab(...lab), labFromRgb, ColorEps);
        });
    });

    describe("lab <-> tanh", () => {
        test.each(LabTestData)
        ("$lab", ({lab}) => {
            const tanh = ColorUtils.labToTanh(...lab);
            const labFromTanh = ColorUtils.tanhToLab(...tanh);

            ArrayUtils.arrayCloseTo(ColorUtils.clampLab(...lab), labFromTanh, ColorEps);
        });
    });

    test("batch color space transform", () => {
        const data = [
            255, 50, 48, 255,
            45, 39, 0, 255,
            0, 255, 255, 255,
            255, 0, 255, 255,
        ];

        const out = ColorUtils.transformColorSpace(ColorUtils.rgbToLab, data, 4);
        ColorUtils.transformColorSpace(ColorUtils.labToTanh, out, 4, out);
        ColorUtils.transformColorSpace(ColorUtils.tanhToLab, out, 4, out);
        ColorUtils.transformColorSpace(ColorUtils.labToRgb, out, 4, out);

        ArrayUtils.arrayCloseTo(data, out, ColorEps);
    });
});