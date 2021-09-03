import * as matrix from "./matrix";

function uniformRandom(from = 0, to = 1): number {
    const dist = to - from;
    return from + Math.random() * dist;
}

function normalRandom(from = 0, to = 1): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while (v === 0) v = Math.random();
    const value = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);

    const dist = to - from;
    return from + value * dist;
}

function rndVector(fn: () => number, size: number, limit: number): matrix.Matrix1D {
    return matrix.fill(() => uniformRandom(-limit, limit), size);
}

function zeroInitializer(size: number, _: number): matrix.Matrix1D {
    return matrix.zero(size);
}

function xavierInitializer(size: number, prevSize: number): matrix.Matrix1D {
    const limit = Math.sqrt(6 / (size + prevSize));
    return rndVector(uniformRandom, size, limit);
}

function xavierNormalInitializer(size: number, prevSize: number): matrix.Matrix1D {
    const limit = Math.sqrt(6 / (size + prevSize));
    return rndVector(normalRandom, size, limit);
}

function uniformInitializer(size: number, _: number): matrix.Matrix1D {
    return rndVector(uniformRandom, size, 1);
}

function normalInitializer(size: number, _: number): matrix.Matrix1D {
    return rndVector(normalRandom, size, 1);
}

export type InitializerT = "xavier" | "xavier_normal" | "uniform" | "normal" | "zero";

export const Initializers = {
    zero: zeroInitializer,
    xavier: xavierInitializer,
    xavier_normal: xavierNormalInitializer,
    uniform: uniformInitializer,
    normal: normalInitializer,
};