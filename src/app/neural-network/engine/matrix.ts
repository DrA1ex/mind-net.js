import * as iter from './iter';

export type Matrix1D = number[];
export type Matrix2D = number[][];

export type OptMatrix1D = number[] | undefined;


export function fill<T>(value_fn: (i: number) => T, length: number): T[] {
    return Array.from(iter.map(iter.range(0, length), value_fn));
}

export function fill_value<T>(value: T, length: number): T[] {
    return fill(() => value, length);
}

export function matrix1d_binary_in_place_op(dst: Matrix1D, b: Matrix1D, op: (x1: number, x2: number) => number) {
    const length = Math.min(dst.length, b.length);

    for (let i = 0; i < length; ++i) {
        dst[i] = op(dst[i], b[i]);
    }

    return dst;
}

export function matrix1d_binary_op(a: Matrix1D, b: Matrix1D, op: (x1: number, x2: number) => number, dst: OptMatrix1D = undefined): Matrix1D {
    const length = Math.min(a.length, b.length);
    const result = dst || new Array(length);

    for (let i = 0; i < length; ++i) {
        result[i] = op(a[i], b[i]);
    }

    return result;
}

export function matrix1d_unary_op(a: Matrix1D, op: (x1: number, i: number) => number, dst: OptMatrix1D = undefined): Matrix1D {
    const length = a.length;
    const result = dst || new Array(length);

    for (let i = 0; i < length; ++i) {
        result[i] = op(a[i], i);
    }

    return result;
}

export function matrix1d_unary_in_place_op(a: Matrix1D, op: (x1: number, i: number) => number): Matrix1D {
    const length = a.length;
    for (let i = 0; i < length; ++i) {
        a[i] = op(a[i], i);
    }

    return a;
}

export function sub(a: Matrix1D, b: Matrix1D, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 - x2, dst);
}

export function add(a: Matrix1D, b: Matrix1D, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 + x2, dst);
}

export function add_to(dst: Matrix1D, b: Matrix1D) {
    matrix1d_binary_in_place_op(dst, b, (x1, x2) => x1 + x2);
}

export function add_scalar(a: Matrix1D, value: number, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_unary_op(a, x1 => x1 + value, dst);
}

export function mul(a: Matrix1D, b: Matrix1D, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 * x2, dst);
}

export function mul_to(dst: Matrix1D, b: Matrix1D): Matrix1D {
    return matrix1d_binary_op(dst, b, (x1, x2) => x1 * x2, dst);
}

export function mul_scalar(a: Matrix1D, value: number, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_unary_op(a, x1 => x1 * value, dst);
}

export function div(a: Matrix1D, b: Matrix1D, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 / x2, dst);
}

export function dot(a: Matrix1D, b: Matrix1D): number {
    const length = Math.min(a.length, b.length);
    let sum = 0, c = 0;

    for (let i = 0; i < length; ++i) {
        const y = a[i] * b[i] - c;
        const t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}


export function zero(length: number): Matrix1D {
    return fill_value(0, length);
}

export function zero_2d(rows: number, cols: number): Matrix2D {
    return fill(() => zero(cols), rows);
}

export function one(length: number): Matrix1D {
    return fill_value(1, length);
}

export function random(length: number, from: number = 0, to: number = 1): Matrix1D {
    const dist = to - from;
    return fill(() => from + Math.random() * dist, length);
}

export function copy(a: Matrix1D): Matrix1D {
    return [...a];
}

export function copy_2d(a: Matrix2D): Matrix2D {
    return a.map(copy);
}

export function max(a: Matrix1D, abs = false): number {
    let max = Number.MIN_VALUE;
    for (let value of a) {
        value = abs ? Math.abs(value) : value;
        if (value > max) {
            max = value;
        }
    }

    return max;
}

export function sum(a: Matrix1D): number {
    let sum = 0;
    for (let value of a) {
        sum += value;
    }

    return sum;
}

export function transform(m: Matrix2D): Matrix2D {
    if (m.length === 0) {
        return [];
    }

    const result = new Array(m[0].length);
    for (let i = 0; i < result.length; i++) {
        result[i] = new Array(m.length);
        for (let j = 0; j < m.length; j++) {
            result[i][j] = m[j][i];
        }
    }

    return result;
}

export function random_2d(rows: number, cols: number, from: number = 0, to: number = 1): Matrix2D {
    return fill(i => random(cols, from, to), rows);
}


export function dot_2d(x1: Matrix2D, x2: Matrix1D, dst: OptMatrix1D = undefined): Matrix1D {
    const length = x1.length;
    const result = dst || new Array(length);
    for (let i = 0; i < length; i++) {
        result[i] = dot(x1[i], x2);
    }

    return result;
}

export function dot_2d_translated(x1: Matrix2D, x2: Matrix1D): Matrix1D {
    const result = new Array(x1[0].length);
    const rowsLength = Math.min(x1.length, x2.length);

    for (let i = 0; i < result.length; ++i) {
        let sum = 0, c = 0;

        for (let j = 0; j < rowsLength; j++) {
            const y = x1[j][i] * x2[j] - c;
            const t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        result[i] = sum;
    }

    return result;
}
