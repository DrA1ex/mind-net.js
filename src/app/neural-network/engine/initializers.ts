import * as matrix from "../../utils/matrix";

function zeroInitializer(size: number, _: number): matrix.Matrix1D {
    return matrix.zero(size);
}

function xavierInitializer(size: number, prevSize: number): matrix.Matrix1D {
    const limit = Math.sqrt(6 / (size + prevSize));
    return matrix.random(size, -limit, limit);
}

function uniformInitializer(size: number, _: number): matrix.Matrix1D {
    return matrix.random(size, -1, 1);
}

export type InitializerT = "xavier" | "uniform" | "zero";

export const Initializers = {
    zero: zeroInitializer,
    xavier: xavierInitializer,
    uniform: uniformInitializer
};