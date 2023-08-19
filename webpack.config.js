import path from "path";
import url from "url";

import nodeExternals from 'webpack-node-externals';

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

export default {
    mode: "production",
    externals: [nodeExternals({
        allowlist: ["tslib"],
        importType: "module",
    })],
    externalsPresets: {
        node: true
    },
    entry: {
        "main": {
            import: [
                "./src/app/neural-network/neural-network.ts",
            ]
        },
    },
    optimization: {
        minimize: false
    },
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
            },
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
    },
    experiments: {
        outputModule: true,
    },
    output: {
        path: path.resolve(__dirname, "lib"),
        filename: '[name].js',
        library: {
            type: "module"
        },
        chunkFormat: "module",
        module: true,
    },
};