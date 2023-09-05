import {Activations} from "../src/app/neural-network/neural-network";

import {
    LeakyReluActivation, LinearActivation, ReluActivation, SigmoidActivation, SoftMaxActivation, TanhActivation
} from "../src/app/neural-network/neural-network";

const ActivationValueTestInput = [-10, -1, -0.5, 0, 0.5, 1, 10];

describe("Activation.value", () => {
    test.each([
        {type: SigmoidActivation, expected: [0.000045397868702434395, 0.2689414213699951, 0.3775406687981454, 0.5, 0.6224593312018546, 0.7310585786300049, 0.9999546021312976]},
        {type: ReluActivation, expected: [-0, -0, -0, 0, 0.5, 1, 10,]},
        {type: LeakyReluActivation, expected: [-3, -0.3, -0.15, 0, 0.5, 1, 10]},
        {type: TanhActivation, expected: [-0.9999999958776927, -0.7615941559557649, -0.46211715726000974, 0, 0.46211715726000974, 0.7615941559557649, 0.9999999958776927]},
        {type: LinearActivation, expected: [-10, -1, -0.5, 0, 0.5, 1, 10]},
        {type: SoftMaxActivation, expected: [2.060560383446606e-9, 0.000016696893724904762, 0.000027528523838869978, 0.00004538686280412047, 0.0000748302861155019, 0.00012337428441120446, 0.9997121810885449]}
    ])("$type", (
        {type, expected}
    ) => {
        const obj = new type();
        expect(obj.forward(ActivationValueTestInput)).toStrictEqual(expected);
    });
});

describe("Activation.moment", () => {
    test.each([
        {type: SigmoidActivation, expected: [0.000045395807735951673, 0.19661193324148185, 0.2350037122015945, 0.25, 0.2350037122015945, 0.19661193324148185, 0.000045395807735907655]},
        {type: ReluActivation, expected: [0, 0, 0, 1, 1, 1, 1]},
        {type: LeakyReluActivation, expected: [0.3, 0.3, 0.3, 1, 1, 1, 1]},
        {type: TanhActivation, expected: [8.244614546626394e-9, 0.41997434161402614, 0.7864477329659274, 1, 0.7864477329659274, 0.41997434161402614, 8.244614546626394e-9]},
        {type: LinearActivation, expected: [1, 1, 1, 1, 1, 1, 1]},
        {type: SoftMaxActivation, expected: [1, 1, 1, 1, 1, 1, 1]}
    ])("$type", (
        {type, expected}
    ) => {
        const obj = new type();
        expect(obj.backward(ActivationValueTestInput)).toStrictEqual(expected);
    });
});

describe("Should use destination array if specified ::value", () => {
    test.each(
        Object.values(Activations)
    )("%p", (type) => {
        const obj = new type();

        const dst = new Array(ActivationValueTestInput.length);
        const withDst = obj.forward(ActivationValueTestInput, dst);
        const withoutDst = obj.forward(ActivationValueTestInput);

        expect(withDst).toStrictEqual(withoutDst);
        expect(withDst).toBe(dst);
    });
});

describe("Should use destination array if specified ::moment", () => {
    test.each(
        Object.values(Activations)
    )("%p", (type) => {
        const obj = new type();

        const dst = new Array(ActivationValueTestInput.length);
        const withDst = obj.backward(ActivationValueTestInput, dst);
        const withoutDst = obj.backward(ActivationValueTestInput);

        expect(withDst).toStrictEqual(withoutDst);
        expect(withDst).toBe(dst);
    });
});