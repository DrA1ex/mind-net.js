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
