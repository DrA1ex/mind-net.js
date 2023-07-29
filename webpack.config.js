import path from "path";
import url from "url";
import {TsconfigPathsPlugin} from "tsconfig-paths-webpack-plugin";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

export default {
    mode: "production",
    entry: {
        "main": {
            import: [
                "./src/app/neural-network/neural-network.ts",
            ]
        },
    },
    devtool: 'source-map',
    experiments: {
        outputModule: true,
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
        plugins: [new TsconfigPathsPlugin({configFile: "./tsconfig.lib.json"})]
    },
    output: {
        path: path.resolve(__dirname, "lib"),
        filename: '[name].js',
        library: {
            type: "module"
        }
    },
};