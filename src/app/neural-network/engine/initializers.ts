import * as matrix from "./matrix";

function zeroInitializer(size: number, _: number): matrix.Matrix1D {
    return matrix.zero(size);
}

function heInitializer(size: number, prevSize: number): matrix.Matrix1D {
    const limit = Math.sqrt(2 / (prevSize + size));
    return matrix.random_1d(size, -limit, limit);
}

function heNormalInitializer(size: number, _: number): matrix.Matrix1D {
    const limit = Math.sqrt(2 / size);
    return matrix.random_normal_1d(size, -limit, limit);
}

function xavierInitializer(size: number, prevSize: number): matrix.Matrix1D {
    const limit = Math.sqrt(6 / (size + prevSize));
    return matrix.random_1d(size, -limit, limit);
}

function xavierNormalInitializer(size: number, prevSize: number): matrix.Matrix1D {
    const limit = Math.sqrt(6 / (size + prevSize));
    return matrix.random_normal_1d(size, -limit, limit);
}

function uniformInitializer(size: number, _: number): matrix.Matrix1D {
    return matrix.random_1d(size, -1, 1);
}

function normalInitializer(size: number, _: number): matrix.Matrix1D {
    return matrix.random_normal_1d(size, -1, 1);
}

export type InitializerT = "he" | "he_normal" | "xavier" | "xavier_normal" | "uniform" | "normal" | "zero";

export const Initializers = {
    he: heInitializer,
    he_normal: heNormalInitializer,
    zero: zeroInitializer,
    xavier: xavierInitializer,
    xavier_normal: xavierNormalInitializer,
    uniform: uniformInitializer,
    normal: normalInitializer,
};