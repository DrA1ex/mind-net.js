import {Initializers} from "../src/app/neural-network/neural-network";
import {MockRandom} from "./mock/common";

const InitializerSize = 5;
const InitializerRandomValues = [
    0.1, 0.3, 0.2, 0.99, 0.5,
    0.4, 0.01, 0.95, 0.4, 0.15
];

let randomMock: jest.SpiedFunction<() => number>;
beforeEach(() => (randomMock = MockRandom(InitializerRandomValues)));
afterEach(() => randomMock.mockRestore());

describe("Should generate correct data", () => {
    test.each(
        [
            {name: "he", expected: [-0.5059644256269407, -0.2529822128134704, -0.3794733192202055, 0.6198064213930025, 0]},
            {name: "he_normal", expected: [-1.4712686189773108, 1.632471813807812, -1.8373398842972768, 3.01847993788442, 0.3740364541732947]},
            {name: "zero", expected: [0, 0, 0, 0, 0]},
            {name: "xavier", expected: [-0.8763560920082657, -0.4381780460041329, -0.6572670690061992, 1.0735362127101253, 0]},
            {name: "xavier_normal", expected: [-2.548311999650398, 2.8275241234392507, -3.182366030375606, 5.228160614043164, 0.6478501425110543]},
            {name: "uniform", expected: [-0.8, -0.4, -0.6, 0.98, 0]},
            {name: "normal", expected: [-2.3262799429493666, 2.5811645738294993, -2.905089435124817, 4.77263583761917, 0.5914035615604016]},
        ]
    )("$name", ({name, expected}) => {
        const initializer = Initializers[name];
        expect(initializer).toBeDefined();

        const generated = initializer(InitializerSize, 0);
        expect(generated).toHaveLength(InitializerSize);
        expect(generated).toStrictEqual(expected);
    });
});

describe("Should generate data with desired size", () => {
    describe.each([Object.keys(Initializers)])
    ("initializer: %s", (name) => {
        describe.each([0, 1, 2, 5])
        ("size: %d", (size) => {
            test.each([0, 1, 2, 5])
            ("prevSize: %d", (prevSize) => {
                const initializer = Initializers[name];
                expect(initializer(size, prevSize)).toHaveLength(size);
            });
        });
    });
});