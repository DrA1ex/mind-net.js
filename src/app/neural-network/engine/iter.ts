export function* range(from: number, to: number): Iterable<number> {
    for (let i = from; i < to; ++i) {
        yield i;
    }
}

export function* map<T, R>(input: Iterable<T>, map: (arg: T, i?: number) => R): Iterable<R> {
    let i = 0;
    for (const item of input) {
        yield map(item, i);
        i += 1;
    }
}

export function* reverse<T>(input: Iterable<T>): Iterable<T> {
    const cache = [];
    for (const e of input) {
        cache.push(e);
    }

    for (let i = cache.length - 1; i >= 0; i--) {
        yield cache[i];
    }
}


export function shuffle<T>(array: Array<T>): Array<T> {
    let currentIndex = array.length, randomIndex;
    while (0 !== currentIndex) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }

    return array;
}

export function zip<T1, T2>(a: Array<T1>, b: Array<T2>): Array<[T1, T2]> {
    const length = Math.max(a.length, b.length);
    const result = new Array(length);

    for (let i = 0; i < length; i++) {
        result[i] = [a[i], b[i]];
    }

    return result;
}

export function* partition<T>(data: T[], size: number): Iterable<T[]> {
    let processed = 0;
    while (processed < data.length) {
        yield data.slice(processed, processed + size);

        processed += size;
    }
}

export function aggregate<T, R>(items: Iterable<T>, fn: (prev: R | undefined, current: T) => R, initial: R | undefined = undefined): R {
    let prev = initial;
    for (const item of items) {
        prev = fn(prev, item);
    }

    return prev!;
}

export function sum(items: Iterable<number>): number {
    return aggregate(items, (p, c) => p! + c, 0);
}

export function max(items: Iterable<number>): number {
    return aggregate(items, (p, c) => Math.max(p!, c), 0);
}

export function fill<T>(value_fn: (i: number) => T, length: number): T[] {
    return Array.from(map(range(0, length), value_fn));
}

export function fill_value<T>(value: T, length: number): T[] {
    return fill(() => value, length);
}

export function fill_random(from: number, to: number, length: number): number[] {
    const dist = to - from;
    return fill(() => from + Math.random() * dist, length);
}

export function* chain<T>(...lists: Iterable<T>[]): Iterable<T> {
    for (const list of lists) {
        for (const listElement of list) {
            yield listElement
        }
    }
}
