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

export function arrayCloseTo(a: ArrayT, b: ArrayT, eps = 1e-8) {
    const aWithEps = dropArrayEps(a, eps);
    const bWithEps = dropArrayEps(b, eps);

    expect(aWithEps).toStrictEqual(bWithEps);
}

export function arrayCloseTo_2d(a: ArrayT[], b: ArrayT[], eps = 1e-8) {
    const aWithEps = a.map(arr => dropArrayEps(arr, eps));
    const bWithEps = b.map(arr => dropArrayEps(arr, eps))

    expect(aWithEps).toStrictEqual(bWithEps);
}