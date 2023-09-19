type Chunk = { buffer: ArrayBuffer; start: number; end: number; size: number; };

export type TypedArrayExt = {
    readonly BYTES_PER_ELEMENT: number;
}

export type TypedArray = (Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array);
export type TypedArrayT<T extends TypedArray> = TypedArrayExt & (new (...args: any[]) => T);

export class ChunkedArrayBuffer {
    chunks: Chunk[] = [];
    bytesOffset = 0;
    byteLength = 0;

    constructor(chunks: ArrayBuffer[], bytesOffset = 0, byteLength = -1) {
        for (const chunk of chunks) {
            this._addChunk(chunk);
        }

        byteLength = byteLength > 0 ? byteLength : this.byteLength;
        this.bytesOffset = Math.max(0, Math.min(bytesOffset, this.chunks[0].size));
        this.byteLength = Math.max(0, Math.min(this.byteLength - this.bytesOffset, byteLength));
    }

    _addChunk(chunk: ArrayBuffer) {
        if (chunk.byteLength === 0) {
            return;
        }

        const lastOffset = this.byteLength;
        this.chunks.push({
            buffer: chunk,
            start: lastOffset,
            end: lastOffset + chunk.byteLength,
            size: chunk.byteLength
        });

        this.byteLength += chunk.byteLength;
    }

    slice(bytesOffset: number, byteLength: number): ChunkedArrayBuffer {
        const startBorder = this.bytesOffset + bytesOffset;
        const endBorder = Math.min(bytesOffset + byteLength, this.byteLength);
        const chunks = this.chunks.filter(c => startBorder < c.end && endBorder >= c.start);

        if (chunks.length > 0) {
            return new ChunkedArrayBuffer(chunks.map(c => c.buffer), this.bytesOffset + bytesOffset - chunks[0].start, byteLength)
        }

        return new ChunkedArrayBuffer([], 0, 0);
    }

    createTypedArray<T extends TypedArray>(type: TypedArrayT<T>, bytesOffset = 0, count = -1): T {
        const itemSize = type.BYTES_PER_ELEMENT;
        const totalSize = count > 0 ? count : Math.floor((this.byteLength - bytesOffset) / itemSize);
        return this.slice(bytesOffset, totalSize * itemSize).toTypedArray(type);
    }

    toTypedArray<T extends TypedArray>(type: TypedArrayT<T>): T {
        const itemSize = type.BYTES_PER_ELEMENT;
        if (this.chunks.length === 0 || this.byteLength < itemSize) {
            return new type();
        }

        const itemsCount = Math.floor(this.byteLength / itemSize);
        if (this.chunks.length === 1 && this.bytesOffset % itemSize === 0) {
            const {buffer} = this.chunks[0];
            return new type(buffer, this.bytesOffset, itemsCount);
        }

        const result = new Uint8Array(itemsCount * itemSize);

        let remaining = result.length;
        let written = 0;
        for (let i = 0; i < this.chunks.length && remaining > 0; i++) {
            const {buffer, size} = this.chunks[i];
            const chunkBytes = new Uint8Array(buffer);

            const offset = i === 0 ? this.bytesOffset : 0;
            const copyLength = Math.min(remaining, size - offset);

            result.set(chunkBytes.subarray(offset, offset + copyLength), written);

            written += copyLength;
            remaining -= copyLength;
        }

        return new type(result.buffer);
    }
}