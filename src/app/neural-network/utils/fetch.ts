import {ChunkedArrayBuffer} from "./array-buffer";

interface IAsyncReader {
    readonly size: number;
    [Symbol.asyncIterator](): AsyncGenerator<Uint8Array>;
}

export type ProgressFn = (read: number, total: number) => void;

export class ObservableStreamLoader {
    static CHUCK_SIZE = 1024 * 1024 * 64;

    stream: IAsyncReader;
    progressFn: ProgressFn;

    constructor(asyncReader: IAsyncReader, progressFn: ProgressFn) {
        this.stream = asyncReader;
        this.progressFn = progressFn;
    }

    async load() {
        const size = this.stream.size;
        const data = new Uint8Array(size);

        this.progressFn(0, size);

        let offset = 0;
        for await (const chunk of this.stream) {
            data.set(chunk, offset);
            offset += chunk.length;

            this.progressFn(offset, size);
        }

        return data.buffer;
    }

    async loadChunked(chunkSize = ObservableStreamLoader.CHUCK_SIZE): Promise<ChunkedArrayBuffer> {
        const totalSize = this.stream.size;
        this.progressFn(0, totalSize);

        const bigChunks = [];
        let totalRead = 0;
        let read = 0;
        let readChunks = [];
        for await (const chunk of this.stream) {
            readChunks.push(chunk.buffer)
            read += chunk.length;
            totalRead += chunk.length;
            if (read >= chunkSize) {
                bigChunks.push(new ChunkedArrayBuffer(readChunks).toTypedArray(Uint8Array).buffer);
                readChunks = [];
                read = 0;
            }

            this.progressFn(totalRead, totalSize);
        }

        if (readChunks.length > 0) {
            bigChunks.push(new ChunkedArrayBuffer(readChunks).toTypedArray(Uint8Array).buffer);
            this.progressFn(totalSize, totalSize);
        }

        return new ChunkedArrayBuffer(bigChunks);
    }
}

export class FetchDataAsyncReader implements IAsyncReader {
    public readonly size: number;

    constructor(public response: Response) {
        if (!response.ok || !response.body) throw new Error("Response should be successful response with body");

        const contentLength = response.headers.get('Content-Length');
        this.size = contentLength ? Number.parseInt(contentLength!) : -1;
    }

    async* [Symbol.asyncIterator]() {
        const reader = this.response.body!.getReader();
        while (true) {
            const chunk = await reader.read();
            if (chunk.done) {
                break;
            }

            yield chunk.value;
        }
    }
}


export class FileAsyncReader implements IAsyncReader {
    public readonly size: number

    constructor(public readonly file: File) {
        this.file = file;
        this.size = file.size;
    }

    async* [Symbol.asyncIterator]() {
        const reader = this.file.stream().getReader()

        while (true) {
            const chunk = await reader.read();
            if (chunk.done) {
                break;
            }

            yield chunk.value;
        }
    }
}