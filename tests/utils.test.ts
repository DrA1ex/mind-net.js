import {jest} from "@jest/globals";

import {FetchDataAsyncReader, Matrix, ProgressUtils} from "../src/app/neural-network/neural-network";
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