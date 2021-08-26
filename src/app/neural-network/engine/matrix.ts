import {GlobalPool, IMemorySlice, ManagedMemorySlice} from "./memory";
import * as iter from './iter';


export type Matrix1D = IMemorySlice;
export type Matrix2D = IMemorySlice[];

export type ManagedMatrix1D = ManagedMemorySlice;
export type ManagedMatrix2D = ManagedMemorySlice[];

export function fill(value_fn: (i: number) => number, length: number): ManagedMatrix1D {
    const result = GlobalPool.alloc(length);
    let i = 0;
    for (const item of iter.map(iter.range(0, length), value_fn)) {
        result.data[i] = item;
        i += 1;
    }

    return result;
}

export function fill_value(value: number, length: number): ManagedMatrix1D {
    return fill(() => value, length);
}

export function matrix1d_binary_in_place_op(a: Matrix1D, b: Matrix1D, op: (x1: number, x2: number) => number) {
    const length = Math.min(a.length, b.length);

    for (let i = 0; i < length; ++i) {
        a.data[i] = op(a.data[i], b.data[i]);
    }
}


export function matrix1d_binary_op(a: Matrix1D, b: Matrix1D, op: (x1: number, x2: number) => number): ManagedMatrix1D {
    const length = Math.min(a.length, b.length);
    const result = GlobalPool.alloc(length);

    for (let i = 0; i < length; ++i) {
        result.data[i] = op(a.data[i], b.data[i]);
    }

    return result;
}

export function matrix1d_unary_op(a: Matrix1D, op: (x1: number, i: number) => number): ManagedMatrix1D {
    const length = a.length;
    const result = GlobalPool.alloc(length);

    for (let i = 0; i < length; ++i) {
        result.data[i] = op(a.data[i], i);
    }

    return result;
}

export function sub(a: Matrix1D, b: Matrix1D): ManagedMatrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 - x2);
}

export function add(a: Matrix1D, b: Matrix1D): ManagedMatrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 + x2);
}

export function add_to(dst: Matrix1D, b: Matrix1D) {
    matrix1d_binary_in_place_op(dst, b, (x1, x2) => x1 + x2);
}

export function add_scalar(a: Matrix1D, value: number): ManagedMatrix1D {
    return matrix1d_unary_op(a, x1 => x1 + value);
}

export function mul(a: Matrix1D, b: Matrix1D): ManagedMatrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 * x2);
}

export function mul_scalar(a: Matrix1D, value: number): ManagedMatrix1D {
    return matrix1d_unary_op(a, x1 => x1 * value);
}

export function div(a: Matrix1D, b: Matrix1D): ManagedMatrix1D {
    return matrix1d_binary_op(a, b, (x1, x2) => x1 / x2);
}

export function dot(a: Matrix1D, b: Matrix1D): number {
    const length = Math.min(a.length, b.length);
    let sum = 0, c = 0;

    for (let i = 0; i < length; ++i) {
        const y = a.data[i] * b.data[i] - c;
        const t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}


export function zero(length: number): ManagedMatrix1D {
    return fill_value(0, length);
}

export function one(length: number): ManagedMatrix1D {
    return fill_value(1, length);
}

export function random(length: number, from: number = 0, to: number = 1): ManagedMatrix1D {
    const dist = to - from;
    return fill(() => from + Math.random() * dist, length);
}

export function copy(a: Matrix1D): ManagedMatrix1D {
    const result = GlobalPool.alloc(a.length);
    for (let i = 0; i < result.length; i++) {
        result.data[i] = a.data[i];
    }
    return result;
}

export function max(a: Matrix1D, abs = false): number {
    let max = Number.MIN_VALUE;
    for (let value of a.data) {
        value = abs ? Math.abs(value) : value;
        if (value > max) {
            max = value;
        }
    }

    return max;
}

export function sum(a: Matrix1D): number {
    let sum = 0;
    for (let value of a.data) {
        sum += value;
    }

    return sum;
}

export function transform(m: Matrix2D): ManagedMatrix2D {
    if (m.length === 0) {
        return [];
    }

    const result: ManagedMatrix1D[] = new Array(m[0].length);
    for (let i = 0; i < result.length; i++) {
        result[i] = GlobalPool.alloc(m.length);
        for (let j = 0; j < m.length; j++) {
            result[i].data[j] = m[j].data[i];
        }
    }

    return result;
}

export function dot_2d(x1: Matrix2D, x2: Matrix1D): ManagedMatrix1D {
    return fill(i => dot(x1[i], x2), x1.length);
}

export function dot_2d_translated(x1: Matrix2D, x2: Matrix1D): ManagedMatrix1D {
    const result = GlobalPool.alloc(x1[0].length);
    const rowsLength = Math.min(x1.length, x2.length);

    for (let i = 0; i < result.length; ++i) {
        let sum = 0, c = 0;

        for (let j = 0; j < rowsLength; j++) {
            const y = x1[j].data[i] * x2.data[j] - c;
            const t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        result.data[i] = sum;
    }

    return result;
}
