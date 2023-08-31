import path from "path";
import url from "url";

import nodeExternals from 'webpack-node-externals';

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

const common = {
    mode: "production",
    externalsPresets: {
        node: true
    },
    optimization: {
        minimize: false,
    },
    experiments: {
        outputModule: true,
    },
}

const commonOutput = {
    path: path.resolve(__dirname, "lib"),
    chunkFormat: "module",
    library: {
        type: "module"
    },
    module: true,
}

const main = {
    ...common,
    name: "main",
    entry: {
        "main": "./src/app/neural-network/neural-network.ts"
    },
    externals: [nodeExternals({
        allowlist: ["tslib"],
        importType: "module",
    })],
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: {
                    loader: 'ts-loader',
                    options: {
                        configFile: "tsconfig.lib.json"
                    }
                },
                exclude: /node_modules/,
            }
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
    },
    output: {
        ...commonOutput,
        filename: '[name].js',
        clean: {
            keep: /package\.json/
        },
        assetModuleFilename: (pathData) => {
            const {filename} = pathData;

            if (filename.endsWith('.ts')) {
                return '[name].js';
            } else {
                return '[name][ext]';
            }
        },
    },
}

const worker = {
    ...common,
    entry: [
        "./lib/parallel.worker.js"
    ],
    dependencies: ["main"],
    output: {
        ...commonOutput,
        filename: "parallel.worker.js"
    },
    externals: {
        "./parallel.worker.impl": "./main.js",
    }
}

export default [main, worker];