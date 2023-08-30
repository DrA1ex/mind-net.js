export enum Color {
    red = "\u001B[31m",
    green = "\u001B[32m",
    yellow = "\u001B[33m",
    blue = "\u001B[34m",
    magenta = "\u001B[35m",
    cyan = "\u001B[36m",
    lightgray = "\u001B[37m",
    default = "\u001B[39m",
    white = "\u001B[97m",
    black = "\u001B[30m",
    reset = "\u001B[0m"
}

export enum BackgroundColor {
    black = "\u001B[40m",
    red = "\u001B[41m",
    green = "\u001B[42m",
    yellow = "\u001B[43m",
    blue = "\u001B[44m",
    magenta = "\u001B[45m",
    cyan = "\u001B[46m",
    lightgray = "\u001B[47m",
    default = "\u001B[49m"
}

export type ProgressOptions = {
    width: number,
    color: Color,
    background: BackgroundColor,
}

const ProgressOptionsDefaults: ProgressOptions = {
    width: 40,
    color: Color.cyan,
    background: BackgroundColor.default,
}

export function* progress(total: number, options: Partial<ProgressOptions> = {}) {
    const opts = {...ProgressOptionsDefaults, ...options};

    const startTime = performance.now();

    let iteration = 0;
    while (iteration < total) {
        const elapsedTime = (performance.now() - startTime) / 1000;
        const speed = iteration > 0 ? elapsedTime / iteration : 0
        const estimateTime = iteration > 0 ? speed * total : 0
        const progress = iteration / total;

        const output = opts.color + opts.background
            + `${Math.floor(progress * 100).toString().padStart(3, " ")}%|`
            + progressBar(progress, opts.width)
            + `| ${iteration}/${total} `
            + `[${formatTime(elapsedTime)}<${formatTime(estimateTime)}, `
            + `${speed.toFixed(2)}s/it]`
            + Color.reset;

        console.log(output);

        yield;
        iteration++;
    }
}

function formatTime(seconds: number) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    seconds = Math.floor(seconds) % 60;

    const times = [hours > 0 ? hours : undefined, minutes, seconds];

    return times.filter(t => t !== undefined)
        .map(t => t!.toString().padStart(2, '0'))
        .join(':');
}

function progressBar(progress: number, width: number) {
    const filledChar = '█';
    const partialChars = ['▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    const emptyChar = ' ';

    // Calculate the number of filled characters and partial character index
    const progressWidth = Math.max(0, Math.min(1, progress)) * width;
    const filledCount = Math.floor(progressWidth);
    const partialIndex = Math.floor(progressWidth % 1 * partialChars.length);

    // Create the progress bar string
    let progressBar = filledChar.repeat(filledCount);

    let partialWidth = 0;
    if (partialIndex > 0) {
        progressBar += partialChars[partialIndex - 1];
        partialWidth = 1;
    }

    progressBar += emptyChar.repeat(width - filledCount - partialWidth);

    return progressBar;
}