export interface IMemorySlice {
    readonly length: number;
    readonly disposed: boolean;

    data: Float32Array;

    free(): void;
}

export class MemorySlice implements IMemorySlice {
    data: Float32Array;
    readonly length: number;
    readonly disposed = false;

    constructor(data: Float32Array) {
        this.data = data;
        this.length = this.data.length
    }

    free(): void {
    }

    static empty() {
        return new MemorySlice(new Float32Array());
    }

    static from(data: Iterable<number>) {
        return new MemorySlice(Float32Array.from(data));
    }

    static alloc(length: number) {
        return new MemorySlice(new Float32Array(length));
    }
}

export class ManagedMemorySlice implements IMemorySlice {
    readonly memory: MemoryPool;
    readonly handle: number[];
    readonly length: number;
    data: Float32Array;

    disposed: boolean = false;

    constructor(memory: MemoryPool, handle: number[], data: Float32Array) {
        this.memory = memory;
        this.handle = handle;
        this.data = data;
        this.length = data.length;
    }

    public free() {
        this.memory.free(this);
    }

    public dispose() {
        this.disposed = true;
    }
}

class Storage {
    memory: MemoryPool;
    id: number;
    data: Float32Array;

    available: number = 0;
    size: number;

    slices: ManagedMemorySlice[] = [];

    constructor(memory: MemoryPool, id: number, size: number) {
        this.memory = memory;
        this.id = id;
        this.size = size;
        this.available = size;
        this.data = new Float32Array(size);
    }

    alloc(length: number): ManagedMemorySlice {
        const start = this.size - this.available;
        const newSlice = new Float32Array(this.data.buffer, start * Float32Array.BYTES_PER_ELEMENT, length)
        const slice = new ManagedMemorySlice(this.memory, [this.id, start, length], newSlice);

        this.slices.push(slice);
        this.available -= slice.length;

        return slice;
    }

    free(slice: ManagedMemorySlice) {
        slice.dispose();

        let deleteFrom = this.slices.length;
        let spaceToFree = 0;
        for (let i = this.slices.length - 1; i >= 0; i--) {
            const current = this.slices[i];
            if (!current.disposed) {
                break;
            }

            deleteFrom = i;
            spaceToFree += current.length;
        }

        if (spaceToFree > 0) {
            this.slices.splice(deleteFrom, this.slices.length - deleteFrom);
            this.available += spaceToFree;
        }
    }
}

export class MemoryPool {
    readonly defaultChunkSize;
    readonly storage: Storage[] = [];

    get totalSize(): number {
        return this.storage.reduce((p, c) => p + c.size, 0);
    }

    get allocatedSlices(): number {
        return this.storage.reduce((p, c) => p + c.slices.length, 0);
    }

    get usedSize(): number {
        return this.storage.reduce((p, c) => p + (c.size - c.available), 0);
    }

    get maxAvailableSize(): number {
        return this.storage.reduce((p, c) => Math.max(p, c.available), 0);
    }

    constructor(chunkSize: number = 2048 * 2048) {
        this.defaultChunkSize = chunkSize;
    }

    alloc(length: number): ManagedMemorySlice {
        if (this.maxAvailableSize < length) {
            this.newChunk(Math.max(this.defaultChunkSize, length * 2));
        }

        for (let i = this.storage.length - 1; i >= 0; i--) {
            const chunk = this.storage[i];
            if (chunk.available >= length) {
                return chunk.alloc(length);
            }
        }

        throw new Error("Unable to allocate memory slice");
    }

    allocFrom(items: IMemorySlice): ManagedMemorySlice {
        const result = this.alloc(items.length);
        result.data.set(items.data);

        return result;
    }

    free(slice: ManagedMemorySlice) {
        const id = slice.handle[0];
        if (id >= 0 && id <= this.storage.length) {
            return this.storage[id].free(slice);
        }

        throw new Error(`Invalid slice handle ${slice.handle}`);
    }

    private newChunk(size: number) {
        this.storage.push(new Storage(this, this.storage.length, size));
    }
}

export const GlobalPool = new MemoryPool();