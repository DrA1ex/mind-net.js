import {jest} from '@jest/globals'

export function SetupMockRandom(values: number[], loop = false) {
    let randomValues = values.concat();

    function reset() {
        randomValues = values.concat();
    }

    function _mock() {
        reset();

        return jest.spyOn(Math, "random").mockImplementation(() => {
            let nextValue = randomValues.shift()

            if (nextValue === undefined && loop) {
                reset();
                nextValue = randomValues.shift();
            }

            if (nextValue === undefined) {
                throw new Error("Too many values")
            }

            return nextValue;
        });
    }


    let randomMock: jest.SpiedFunction<() => number>;
    beforeEach(() => {
        randomMock = _mock();
    });

    afterEach(() => {
        randomMock.mockRestore();
    });

    return {reset};
}

export function MockFunctionSequential<T>(values: T[]) {
    const _values = values.concat();

    return jest.fn().mockImplementation((...args: any[]) => {
        const nextValue = _values.shift();
        if (nextValue === undefined) throw new Error("Too many values")

        return structuredClone(nextValue);
    });
}