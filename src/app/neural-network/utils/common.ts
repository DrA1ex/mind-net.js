import * as Matrix from "../engine/matrix";
import {Matrix1D, Matrix2D} from "../engine/matrix";

export function pickRandomItem(trainingData: Matrix.Matrix1D[]) {
    return trainingData[Math.floor(Math.random() * trainingData.length)]
}

export function flatten(input: Matrix2D): Matrix1D {
    const result = [];
    for (const entry of input) {
        result.push(...entry);
    }

    return result;
}

type ReducerFn = (prev: number, current: number) => number;
type MatrixReduceFn = (fn: ReducerFn, initial: number) => number;

export function reduce(input: Matrix1D, fn: (prev: number, current: number) => number, initial: number): number {
    return (input.reduce as MatrixReduceFn)(fn, initial);
}

export function absoluteAccuracy(input: Matrix.Matrix1D[], expected: Matrix.Matrix1D[], k = 2.5) {
    if (!expected) return 0;

    let precision;
    if (expected[0].length > 1) {
        precision = std(expected.map(r => std(r))) / k;
    } else {
        precision = std(flatten(expected)) / k;
    }

    const rows = input.length;
    const columns = input[0].length;

    let outSum = 0;
    for (let i = 0; i < rows; i++) {
        let sum = 0;
        for (let j = 0; j < columns; j++) {
            sum += Math.abs(input[i][j] - expected[i][j])
        }

        outSum += sum / columns < precision ? 1 : 0;
    }

    return outSum / rows;
}

export function std(data: Matrix.Matrix1D) {
    const length = data.length;
    const mean = reduce(data, (p, c) => (p + c) / length, 0);
    const meanOfSquareDiff = reduce(data, (p, c) => p + Math.pow(c - mean, 2) / length, 0);
    return Math.sqrt(meanOfSquareDiff);
}

type Unit = string | string[];
type UnitConfig = { unit: Unit, exp: number, threshold?: number }

export function formatUnitCustom(value: number, unit: string, unitsConfig: UnitConfig[], fractionDigits: number) {
    let sizeUnit: Unit = "";
    for (let i = 0; i < unitsConfig.length; i++) {
        if (value >= unitsConfig[i].exp * (unitsConfig[i].threshold ?? 1)) {
            value /= unitsConfig[i].exp;
            sizeUnit = unitsConfig[i].unit;
            break;
        }
    }

    sizeUnit = formatUnitSuffix(fractionDigits > 0 ? value : Math.round(value), sizeUnit);
    return `${value.toFixed(fractionDigits)} ${sizeUnit}${unit}`;
}

export function formatUnitSuffix(value: number, unit: Unit) {
    if (unit instanceof Array) {
        if (Math.abs(value) === 1) {
            return unit[0];
        } else {
            return unit[1];
        }
    }

    return unit;
}

export function formatUnit(value: number, unit: string, fractionDigits = 2, exp = 1000) {
    const units = [
        {unit: "T", exp: Math.pow(exp, 4)},
        {unit: "G", exp: Math.pow(exp, 3)},
        {unit: "M", exp: Math.pow(exp, 2)},
        {unit: "K", exp: exp},
    ]

    return formatUnitCustom(value, unit, units, fractionDigits);
}

export function formatByteSize(size: number) {
    return formatUnit(size, "B", 2, 1024)
}

export function formatTimeSpan(ms: number, fractionDigits = 2) {
    const units: UnitConfig[] = [
        {unit: "d", exp: 60 * 60 * 24 * 1000},
        {unit: "h", exp: 60 * 60 * 1000, threshold: 3},
        {unit: "m", exp: 60 * 1000, threshold: 5},
        {unit: "s", exp: 1000},
        {unit: "ms", exp: 1},
    ]

    return formatUnitCustom(ms, "", units, fractionDigits);
}