import {jest} from '@jest/globals'

export function MockRandom(values: number[]) {
    const randomValues = values.concat();
    return jest.spyOn(Math, "random").mockImplementation(() => {
        const nextValue = randomValues.shift()
        if (nextValue === undefined) throw new Error("Too many values")

        return nextValue
    });
}