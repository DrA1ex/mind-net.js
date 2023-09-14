import {Matrix1D} from "../engine/matrix";


type ColorSpaceTransformFn = (a: number, b: number, c: number) => [number, number, number];

export function transformColorSpace(
    transformFn: ColorSpaceTransformFn, data: Matrix1D, channels = 3, dst?: Matrix1D
): Matrix1D {
    if (channels !== 3 && channels !== 4) throw new Error(`Unsupported channel count: ${channels}`);
    if (data.length / channels % 1 !== 0) throw new Error("Bad data length");

    const result = dst ?? new Array(data.length);
    for (let offset = 0; offset < data.length; offset += channels) {
        [
            result[offset], result[offset + 1], result[offset + 2]
        ] = transformFn(data[offset], data[offset + 1], data[offset + 2]);

        if (channels === 4) result[offset + 3] = data[offset + 3];
    }

    return result;
}

export function transformChannelCount(data: Matrix1D, inChannels: number, outChannels: number, dst?: Matrix1D): Matrix1D {
    if (inChannels !== 1 && inChannels !== 3 && inChannels !== 4) throw new Error("Unsupported input channels count");
    if (outChannels !== 1 && outChannels !== 3 && outChannels !== 4) throw new Error("Unsupported output channels count");
    if (inChannels === outChannels) throw new Error("Input and output channels count should be different");

    const count = data.length / inChannels;
    if (count % 1 !== 0) throw new Error("Bad data length");

    const outSize = count * outChannels;
    if (dst && dst.length < outSize) throw new Error("Not enough space in dst array");

    function _set4to3(src: Matrix1D, srcOffset: number, dst: Matrix1D, dstOffset: number) {
        dst[dstOffset] = src[srcOffset];
        dst[dstOffset + 1] = src[srcOffset + 1];
        dst[dstOffset + 2] = src[srcOffset + 2];
    }

    function _set3to4(src: Matrix1D, srcOffset: number, dst: Matrix1D, dstOffset: number) {
        dst[dstOffset] = src[srcOffset];
        dst[dstOffset + 1] = src[srcOffset + 1];
        dst[dstOffset + 2] = src[srcOffset + 2];
        dst[dstOffset + 3] = 255;
    }

    function _set3to1(src: Matrix1D, srcOffset: number, dst: Matrix1D, dstOffset: number) {
        // Calculate grayscale value using the luminosity method
        dst[dstOffset] = 0.2989 * src[srcOffset] + 0.587 * src[srcOffset + 1] + 0.114 * src[srcOffset + 2];
    }

    function _set1to(src: Matrix1D, srcOffset: number, dst: Matrix1D, dstOffset: number, dstChannels: number) {
        dst[dstOffset] = src[srcOffset];
        dst[dstOffset + 1] = src[srcOffset];
        dst[dstOffset + 2] = src[srcOffset];
        if (dstChannels === 4) dst[dstOffset + 3] = 255;
    }

    const result = dst ?? new Array(count * outChannels);
    for (let i = 0; i < count; i++) {
        const inOffset = i * inChannels;
        const outOffset = i * outChannels;

        if (inChannels === 4 && outChannels === 3) _set4to3(data, inOffset, result, outOffset);
        else if (inChannels === 3 && outChannels === 4) _set3to4(data, inOffset, result, outOffset);
        else if (inChannels === 1) _set1to(data, inOffset, result, outOffset, outChannels);
        else _set3to1(data, inOffset, result, outOffset);
    }

    return result;

}

export function rgbToTanh(r: number, g: number, b: number): [number, number, number] {
    return [
        Math.max(-1, Math.min(1, r / 127.5 - 1)),
        Math.max(-1, Math.min(1, g / 127.5 - 1)),
        Math.max(-1, Math.min(1, b / 127.5 - 1)),
    ]
}

export function tanhToRgb(r: number, g: number, b: number): [number, number, number] {
    return clampRgb(
        (r + 1) * 127.5,
        (g + 1) * 127.5,
        (b + 1) * 127.5,
    )
}

export function labToTanh(x: number, y: number, z: number): [number, number, number] {
    return [
        Math.max(-1, Math.min(1, x / 50 - 1)),
        Math.max(-1, Math.min(1, y / 128)),
        Math.max(-1, Math.min(1, z / 128)),
    ]
}

export function tanhToLab(x: number, y: number, z: number): [number, number, number] {
    return clampLab(
        (x + 1) * 50,
        y * 128,
        z * 128,
    )
}

export function rgbToLab(red: number, green: number, blue: number): [number, number, number] {
    let [r, g, b] = clampRgb(red, green, blue);

    r = pivotRgbToXyz(r / 255);
    g = pivotRgbToXyz(g / 255);
    b = pivotRgbToXyz(b / 255);

    let x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047;
    let y = (r * 0.2126729 + g * 0.7151522 + b * 0.072175);
    let z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883;

    x = pivotXyzToLab(x);
    y = pivotXyzToLab(y);
    z = pivotXyzToLab(z);

    return clampLab(
        (116 * y) - 16,
        500 * (x - y),
        200 * (y - z),
    );
}

export function labToRgb(l: number, a: number, b: number): [number, number, number] {
    let y = (l + 16) / 116;
    let x = a / 500 + y;
    let z = y - b / 200;

    x = pivotLabToXyz(x) * 0.95047;
    y = pivotLabToXyz(y);
    z = pivotLabToXyz(z) * 1.08883;

    const r = x * 3.24045 + y * -1.53714 + z * -0.498532;
    const g = x * -0.969266 + y * 1.87601 + z * 0.0415561;
    const _b = x * 0.0556434 + y * -0.204026 + z * 1.05723;

    return clampRgb(
        pivotXyzToRgb(r) * 255,
        pivotXyzToRgb(g) * 255,
        pivotXyzToRgb(_b) * 255,
    );
}

function pivotXyzToRgb(value: number): number {
    if (value <= 0.0031308) {
        return value * 12.92;
    }
    return 1.055 * Math.pow(value, 1 / 2.4) - 0.055;
}

function pivotRgbToXyz(value: number): number {
    if (value <= 0.0031308) {
        return value / 12.92;
    }
    return Math.pow((value + 0.055) / 1.055, 2.4);
}

function pivotXyzToLab(value: number): number {
    const epsilon = 0.008856;
    const kappa = 7.787;

    return value > epsilon ? Math.cbrt(value) : kappa * value + 16 / 116;
}

function pivotLabToXyz(value: number): number {
    const epsilon = 0.008856;
    const kappa = 7.787;

    const cValue = value ** 3;
    return cValue > epsilon ? cValue : (value - 16 / 116) / kappa
}

export function clampRgb(r: number, g: number, b: number): [number, number, number] {
    return [
        Math.max(0, Math.min(255, r)),
        Math.max(0, Math.min(255, g)),
        Math.max(0, Math.min(255, b)),
    ]
}

export function clampLab(x: number, y: number, z: number): [number, number, number] {
    return [
        Math.max(0, Math.min(100, x)),
        Math.max(-128, Math.min(128, y)),
        Math.max(-128, Math.min(128, z)),
    ]
}