import path from "path";
import url from "url";

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
    optimization: {
        minimize: false
    },
    experiments: {
        outputModule: true,
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
    output: {
        path: path.resolve(__dirname, "lib"),
        filename: '[name].js',
        library: {
            type: "module"
        }
    },
};