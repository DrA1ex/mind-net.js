export function* range(from: number, to: number): Iterable<number> {
    for (let i = from; i < to; ++i) {
        yield i;
    }
}

export function* map<T, R>(input: Iterable<T>, map: (arg: T, i: number) => R): Iterable<R> {
    let i = 0;
    for (const item of input) {
        yield map(item, i++);
    }
}

export function* map2d<T, R>(input: Iterable<Iterable<T>>, mapFn: (arg: T, i: number, j: number) => R): Iterable<Iterable<R>> {
    let i = 0;
    for (const row of input) {
        yield map(row, (item, j) => mapFn(item, i, j))
        i += 1;
    }
}

export function sum(input: Iterable<number>): number {
    let s = 0;
    for (const item of input) {
        s += item;
    }

    return s;
}

export function max(input: Iterable<number>): number {
    let m = Number.NEGATIVE_INFINITY;
    for (const item of input) {
        if (item > m) {
            m = item;
        }
    }

    return m;
}

export function reverse<T>(input: Iterable<T>): Iterable<T> {
    return Array.from(input).reverse();
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

export function* shuffled<T>(array: Array<T>): Iterable<T> {
    const shuffledIndices = shuffle(Array.from(range(0, array.length)));
    for (const index of shuffledIndices) {
        yield array[index];
    }
}

export function* zip<T1, T2>(a: Array<T1>, b: Array<T2>): Iterable<[T1, T2]> {
    const length = Math.max(a.length, b.length);

    for (let i = 0; i < length; i++) {
        yield [a[i], b[i]];
    }
}

export function* zip_iter<T1, T2>(a: Iterable<T1>, b: Iterable<T2>): Iterable<[T1, T2]> {
    const aIter = a[Symbol.iterator]();
    const bIter = b[Symbol.iterator]();

    let aNext, bNext;
    while (true) {
        aNext = aIter.next();
        bNext = bIter.next();

        if (aNext.done || bNext.done) {
            break;
        }

        yield [aNext.value, bNext.value];
    }
}

export function* join<T1, T2>(a: Iterable<T1>, b: Iterable<T2>): Iterable<T1 | T2> {
    for (const t1 of a) {
        yield t1;
    }

    for (const t2 of b) {
        yield t2;
    }
}

export function* partition<T>(data: Iterable<T>, partitionSize: number): Iterable<T[]> {
    const iterated = iterate(data);
    while (true) {
        const batch = [];
        let cnt = 0;
        for (const item of take(iterated, partitionSize)) {
            batch.push(item);
            cnt++;
        }
        if (cnt > 0) {
            yield batch;
        } else {
            break;
        }
    }
}

export function* take<T>(data: Iterable<T>, size: number): Iterable<T> {
    const iterator = data[Symbol.iterator]();
    for (let cnt = 0; cnt < size; cnt++) {
        const {value, done} = iterator.next();
        if (done) return;

        yield value;
    }
}

export function* iterate<T>(data: Iterable<T>): Iterable<T> {
    for (const item of data) {
        yield item;
    }
}
