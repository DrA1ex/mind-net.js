const AxisSymbol = "┼"
const SpaceSymbol = " ";

const ChartHorizontal = ["─", "╯", "╮",];
const ChartVertical = ["│", "╭", "╰",];

const States = {
    straight: 0,
    up: 1,
    down: 2
}

enum PlotTilePositionFlags {
    top = 0,
    bottom = 1 << 1,
    left = 1 << 4,
    right = 1 << 5,

    top_left = top | left,
    top_right = top | right,
    bottom_left = bottom | left,
    bottom_right = bottom | right,
}

enum Color {
    reset = "\x1b[0m",
    white = "\x1b[97m",
    red = "\x1b[31m",
    green = "\x1b[32m",
    yellow = "\x1b[33m",
    blue = "\x1b[34m",
    magenta = "\x1b[35m",
    cyan = "\x1b[36m",
    lightgray = "\x1b[37m",
    default = "\x1b[39m"
}

type PlotOptions = {
    showAxis: boolean,
    horizontalBoundary: number
    verticalBoundary: number
    title: string,
    titlePosition: PlotTilePositionFlags
}

type PlotSeriesConfig = {
    color: Color
}

export class ConsolePlot {
    public showAxis: boolean;
    public horizontalBoundary: number;
    public verticalBoundary: number;
    public title: string;
    public titlePosition: PlotTilePositionFlags;

    public readonly width;
    public readonly height;

    public readonly screen!: string[][];

    private readonly series: number[][] = [];
    private readonly seriesColors: PlotSeriesConfig[] = [];

    constructor(
        width = 80,
        height = 10, {
            showAxis = true,
            title = "", titlePosition = PlotTilePositionFlags.top,
            horizontalBoundary = 0, verticalBoundary = 1,
        }: Partial<PlotOptions> = {}
    ) {
        this.width = width;
        this.height = height;

        this.showAxis = showAxis;
        this.title = title;
        this.titlePosition = titlePosition;
        this.horizontalBoundary = horizontalBoundary;
        this.verticalBoundary = verticalBoundary;

        this.screen = new Array(this.height);
        for (let i = 0; i < this.height; i++) {
            this.screen[i] = new Array(this.width).fill(SpaceSymbol);
        }
    }

    public addSeries(options: Partial<PlotSeriesConfig> = {}) {
        this.series.push([])
        this.seriesColors.push({...{color: Color.white}, ...options});

        return this.series.length - 1;
    }

    public addSeriesEntry(seriesIndex: number, value: number) {
        if (seriesIndex >= this.series.length) throw new Error("Wrong series index");
        this.series[seriesIndex].push(value);
    }

    public redraw() {
        this._clear();

        let max = Number.NEGATIVE_INFINITY;
        let min = Number.POSITIVE_INFINITY;
        for (const data of this.series) {
            for (const value of data) {
                max = Math.max(max, value);
                min = Math.min(min, value);
            }
        }

        const yStep = Math.max(0.001, max - min) / (this.height - 1 - this.verticalBoundary * 2);
        min -= yStep * this.verticalBoundary;
        max += yStep * this.verticalBoundary;

        let xOffset = 0;
        if (this.showAxis) {
            const labelPadding = this._drawAxis(max, min, yStep);
            xOffset = labelPadding + 2;
        }

        const maxSeriesLength = this.width - xOffset - this.horizontalBoundary * 2 + 1;
        for (let i = 0; i < this.series.length; i++) {
            const data = this._shrink(this.series[i], maxSeriesLength);
            const {color} = this.seriesColors[i];

            if (data.length <= 1) continue;

            let lastState = States.straight;
            let lastY = this._getY(data[0] - min, yStep);

            let x = xOffset + this.horizontalBoundary;
            for (let j = 1; j < data.length; j++) {
                if (!Number.isFinite(data[j])) break;

                const y = this._getY(data[j] - min, yStep)
                const state = y === lastY ? States.straight
                    : y < lastY ? States.up : States.down;

                if (lastState === States.straight) {
                    this.screen[lastY][x++] = color + ChartHorizontal[lastState];
                    this.screen[lastY][x] = color + ChartHorizontal[state];
                } else {
                    this.screen[lastY][x++] = color + ChartVertical[lastState];

                    if (state === States.straight) {
                        this.screen[y][x] = color + ChartHorizontal[state];
                    } else {
                        this.screen[lastY][x] = color + ChartHorizontal[state];
                        this.screen[y][x] = color + ChartVertical[state];
                    }
                }

                if (y !== lastY) {
                    this._fillVertical(color, x, Math.min(y, lastY) + 1, Math.max(y, lastY) - 1);
                }

                lastY = y;
                lastState = state;
            }
        }

        this._drawTitle(xOffset);
    }

    public paint(): string {
        this.redraw();

        return this.screen.map(row => row.join("")).join("\n") + Color.reset;
    }

    private _drawAxis(max: number, min: number, yStep: number) {
        const labelPadding = Math.max(
            Math.abs(max).toFixed(2).length,
            Math.abs(min).toFixed(2).length) + 1;

        for (let i = 0; i < this.height; i++) {
            const index = this.height - 1 - i;
            const axisValue = min + yStep * i;
            const label = axisValue.toFixed(2).padStart(labelPadding, " ");

            for (let j = 0; j < label.length; j++) {
                this.screen[index][j] = Color.white + label[j];
            }

            this.screen[index][labelPadding + 1] = Color.white + AxisSymbol;
        }
        return labelPadding;
    }

    private _drawTitle(plotOffset: number) {
        const maxWidth = this.width - plotOffset;
        if (!this.title || maxWidth <= 4) return;

        let label = this._clipLabel(this.title, maxWidth, this.horizontalBoundary);

        let x;
        if (this.titlePosition & PlotTilePositionFlags.left) {
            x = 0
        } else if (this.titlePosition & PlotTilePositionFlags.right) {
            x = this.width - label.length - 1;
        } else {
            x = plotOffset + Math.round(maxWidth / 2 - label.length / 2)
        }

        let y = 0
        if (this.titlePosition & PlotTilePositionFlags.bottom) {
            y = this.height - 1;
        }

        for (let i = 0; i < label.length; i++) {
            this.screen[y][x + i] = Color.cyan + label[i];
        }
    }

    private _clipLabel(label: string, maxLength: number, boundary: number): string {
        if (label.length > maxLength) {
            return label.slice(0, maxLength - boundary - 1) + "…";
        }

        return label
    }

    private _clear() {
        for (let i = 0; i < this.height; i++) {
            this.screen[i].fill(SpaceSymbol);
        }
    }

    private _shrink(data: number[], maxLength: number) {
        if (data.length <= maxLength) {
            return data;
        }

        const shrunk = new Array(maxLength);
        const scaleFactor = data.length / maxLength;

        for (let i = 0; i < maxLength; i++) {
            const compressedIndex = i * scaleFactor;
            const index1 = Math.floor(compressedIndex);
            const index2 = Math.ceil(compressedIndex);
            const weight2 = compressedIndex - index1;
            const weight1 = 1 - weight2;

            shrunk[i] = (data[index1] * weight1) + (data[index2] * weight2);
        }

        return shrunk;
    }

    private _getY(value: number, yStep: number) {
        return this.height - 1 - Math.max(0, Math.min(this.height - 1, Math.round(value / yStep)));
    }

    private _fillVertical(color: Color, x: number, fromY: number, toY: number) {
        for (let i = fromY; i <= toY; i++) {
            this.screen[i][x] = color + ChartVertical[0];
        }
    }
}

type ChartPlotConfig = {
    xOffset: number,
    yOffset: number,
    width: number,
    height: number,
};

export class MultiPlotChart {
    public width: number = 0;
    public height: number = 0;

    public readonly plots: ConsolePlot[] = [];
    private readonly configs = new Map<ConsolePlot, ChartPlotConfig>();

    public screen!: string[][];

    public addPlot(config: ChartPlotConfig, options?: Partial<PlotOptions>): number {
        const plot = new ConsolePlot(config.width, config.height, options);
        this.plots.push(plot);
        this.configs.set(plot, config);

        let width = 0, height = 0;
        for (const conf of this.configs.values()) {
            width = Math.max(width, conf.xOffset + conf.width);
            height = Math.max(height, conf.yOffset + conf.height);
        }

        this.screen = new Array(height);
        for (let i = 0; i < height; i++) {
            this.screen[i] = new Array(width).fill(SpaceSymbol);
        }

        return this.plots.length - 1;
    }

    public addPlotSeries(plotId: number, config: Partial<PlotSeriesConfig>): number {
        this._assertChartId(plotId);

        return this.plots[plotId].addSeries(config);
    }

    public addSeriesEntry(plotId: number, seriesId: number, entry: number) {
        this._assertChartId(plotId);

        this.plots[plotId].addSeriesEntry(seriesId, entry);
    }

    public redraw() {
        for (const plot of this.plots) {
            this._drawPlot(plot);
        }
    }

    public paint(): string {
        this.redraw();

        return this.screen.map(row => row.join("")).join("\n") + Color.reset;
    }

    private _assertChartId(id: number) {
        if (id >= this.plots.length) {
            throw new Error("Wrong chart id");
        }
    }

    private _drawPlot(plot: ConsolePlot) {
        const config = this.configs.get(plot)!;
        const {xOffset, yOffset, width} = config;

        plot.redraw();
        for (let y = 0; y < plot.screen.length; y++) {
            const row = plot.screen[y];
            for (let x = 0; x < row.length; x++) {
                this.screen[y + yOffset][x + xOffset] = row[x];
            }
        }
    }
}
