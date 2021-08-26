import * as matrix from "./matrix";
import * as iter from "./iter";
import {MemorySlice} from "./memory";

function zeroInitializer(size: number, _: number): matrix.Matrix1D {
    return MemorySlice.from(iter.fill_value(0, size));
}

function xavierInitializer(size: number, prevSize: number): matrix.Matrix1D {
    const limit = Math.sqrt(6 / (size + prevSize));
    return MemorySlice.from(iter.fill_random(-limit, limit, size));
}

function uniformInitializer(size: number, _: number): matrix.Matrix1D {
    return MemorySlice.from(iter.fill_random(-1, 1, size));
}

export type InitializerT = "xavier" | "uniform" | "zero";

export const Initializers = {
    zero: zeroInitializer,
    xavier: xavierInitializer,
    uniform: uniformInitializer
};