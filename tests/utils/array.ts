type ArrayT = number[] | Float32Array | Float64Array;

function dropEps(value: number, eps: number) {
    const result = Math.round(value / eps) * eps
    if (Object.is(result, -0)) return 0;
    return result
}

function dropArrayEps(arr: ArrayT, eps: number) {
    if (!(arr instanceof Array)) arr = Array.from(arr);

    return arr.map(v => dropEps(v, eps));
}

export function arrayCloseTo(actual: ArrayT, expected: ArrayT, eps = 1e-8) {
    const actWithEps = dropArrayEps(actual, eps);
    const expWithEps = dropArrayEps(expected, eps);

    expect(actWithEps).toStrictEqual(expWithEps);
}

export function arrayCloseTo_2d(actual: ArrayT[], expected: ArrayT[], eps = 1e-8) {
    const actWithEps = actual.map(arr => dropArrayEps(arr, eps));
    const expWithEps = expected.map(arr => dropArrayEps(arr, eps))

    expect(actWithEps).toStrictEqual(expWithEps);
}