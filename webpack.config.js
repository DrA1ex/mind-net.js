import path from "path";
import url from "url";
import {TsconfigPathsPlugin} from "tsconfig-paths-webpack-plugin";
import TerserPlugin from "terser-webpack-plugin";

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
    optimization: {
        splitChunks: {chunks: 'all'},
        usedExports: true,
    },
    output: {
        path: path.resolve(__dirname, "lib"),
        filename: '[name].js',
        libraryTarget: 'umd',
        library: 'neural-network',
        umdNamedDefine: true
    },
};