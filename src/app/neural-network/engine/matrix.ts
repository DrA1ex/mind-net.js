import * as iter from './iter';

export type Matrix1D = number[];
export type Matrix2D = number[][];

export type OptMatrix1D = number[] | undefined;
export type OptMatrix2D = number[][] | undefined;


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
    const result = dst ?? new Array(length);

    for (let i = 0; i < length; ++i) {
        result[i] = op(a[i], b[i]);
    }

    return result;
}

export function matrix1d_unary_op(a: Matrix1D, op: (x1: number, i: number) => number, dst: OptMatrix1D = undefined): Matrix1D {
    const length = a.length;
    const result = dst ?? new Array(length);

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

export function matrix2d_unary_in_place_op(a: Matrix2D, op: (x1: number, i: number, j: number) => number): Matrix2D {
    const rows = a.length;
    const columns = a[0].length
    for (let i = 0; i < rows; ++i) {
        for (let j = 0; j < columns; j++) {
            a[i][j] = op(a[i][j], i, j);
        }
    }

    return a;
}

export function matrix2d_binary_in_place_op(dst: Matrix2D, b: Matrix2D, op: (x1: number, x2: number) => number) {
    const rows = Math.min(dst.length, b.length);
    const columns = Math.min(dst[0].length, b[0].length);

    for (let i = 0; i < rows; ++i) {
        for (let j = 0; j < columns; j++) {
            dst[i][j] = op(dst[i][j], b[i][j]);
        }
    }

    return dst;
}

export function matrix2d_binary_op(a: Matrix2D, b: Matrix2D, op: (x1: number, x2: number) => number, dst: OptMatrix2D = undefined) {
    const rows = Math.min(a.length, b.length);
    const columns = Math.min(a[0].length, b[0].length);
    const result = dst ?? zero_2d(rows, columns);

    for (let i = 0; i < rows; ++i) {
        for (let j = 0; j < columns; j++) {
            result[i][j] = op(a[i][j], b[i][j]);
        }
    }

    return result;
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
    return matrix1d_binary_in_place_op(dst, b, (x1, x2) => x1 * x2);
}

export function mul_scalar(a: Matrix1D, value: number, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_unary_op(a, x1 => x1 * value, dst);
}

export function div(a: Matrix1D, b: Matrix1D, dst: OptMatrix1D = undefined): Matrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 / x2, dst);
}

export function copy_to(src: Matrix1D, dst: Matrix1D): Matrix1D {
    for (let i = 0; i < src.length; i++) {
        dst[i] = src[i];
    }

    return dst;
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

function random_n(from = 0, to = 1): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while (v === 0) v = Math.random();
    const value = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);

    const dist = to - from;
    return from + value * dist;
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

export function one_2d(rows: number, cols: number): Matrix2D {
    return fill(() => one(cols), rows);
}

export function random_1d(length: number, from: number = 0, to: number = 1): Matrix1D {
    const dist = to - from;
    return fill(() => from + Math.random() * dist, length);
}

export function random_normal_1d(length: number, from: number = 0, to: number = 1): Matrix1D {
    return fill(() => random_n(from, to), length);
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

export function transpose(m: Matrix2D): Matrix2D {
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
    return fill(() => random_1d(cols, from, to), rows);
}

export function random_normal_2d(rows: number, cols: number, from: number = 0, to: number = 1): Matrix2D {
    return fill(() => random_1d(cols, from, to), rows);
}

export function dot_2d(x1: Matrix2D, x2: Matrix1D, dst: OptMatrix1D = undefined): Matrix1D {
    const length = x1.length;
    const result = dst ?? new Array(length);
    for (let i = 0; i < length; i++) {
        result[i] = dot(x1[i], x2);
    }

    return result;
}

export function dot_2d_translated(x1: Matrix2D, x2: Matrix1D, dst?: Matrix1D): Matrix1D {
    const result = dst ?? new Array(x1[0].length);
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
