import {jest} from '@jest/globals'

export function SetupMockRandom(values: number[]) {
    function _mock() {
        const randomValues = values.concat();
        return jest.spyOn(Math, "random").mockImplementation(() => {
            const nextValue = randomValues.shift()
            if (nextValue === undefined) throw new Error("Too many values")

            return nextValue
        });
    }

    let randomMock: jest.SpiedFunction<() => number>;

    beforeEach(() => {
        randomMock = _mock();
    });

    afterEach(() => {
        randomMock.mockRestore();
    });
}

export function MockFunctionSequential<T>(values: T[]) {
    const _values = values.concat();

    return jest.fn().mockImplementation((...args: any[]) => {
        const nextValue = _values.shift();
        if (nextValue === undefined) throw new Error("Too many values")

        return structuredClone(nextValue);
    });
}