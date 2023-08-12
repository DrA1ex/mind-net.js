import {
    BackgroundColor,
    Color,
    LabelPositionFlags,
    MultiPlotChart,
    PlotAxisScale, PlotSeriesAggregationFn,
    PlotSeriesOverflow
} from "text-graph.js";

import {ModelBase} from "./engine/models/base";
import {Matrix2D} from "./engine/matrix";

type TrainingDashboardOptionsT = {
    width: number,
    height: number
}

const TrainingDashboardOptionsDefaults = {
    width: 140,
    height: 20
}


export class TrainingDashboard {
    public readonly PlotId = {loss: 0, accuracy: 1, lr: 2}

    private _dashboard: MultiPlotChart;
    private set dashboard(value) {this._dashboard = value;}

    public get dashboard() {return this._dashboard;}
    public readonly options: TrainingDashboardOptionsT;

    constructor(
        public readonly model: ModelBase,
        public readonly testInput: Matrix2D,
        public readonly testTrue: Matrix2D,
        options: Partial<TrainingDashboardOptionsT> = {}
    ) {
        this.options = {...TrainingDashboardOptionsDefaults, ...options};
        this._dashboard = this._createDashboard(this.options);
    }

    public update() {
        const {loss, accuracy} = this.model.evaluate(this.testInput, this.testTrue);

        this.dashboard.title = `Epoch: ${this.model.epoch}`;

        this.dashboard.plots[this.PlotId.loss].title = `Loss: ${loss.toFixed(6)}`;
        this.dashboard.addSeriesEntry(this.PlotId.loss, 0, loss);

        this.dashboard.plots[this.PlotId.lr].title = `L. Rate: ${this.model.optimizer.lr.toFixed(6)}`;
        this.dashboard.addSeriesEntry(this.PlotId.lr, 0, this.model.optimizer.lr);

        this.dashboard.plots[this.PlotId.accuracy].title = `Accuracy: ${accuracy.toFixed(2)}`;
        this.dashboard.addSeriesEntry(this.PlotId.accuracy, 0, accuracy);
    }

    public chart(): string {
        return this._dashboard.paint();
    }

    public print() {
        const chart = this.chart();
        console.clear();
        console.log(chart);
    }

    public reset() {
        this.dashboard = this._createDashboard(this.options);
    }

    private _createDashboard(options: TrainingDashboardOptionsT) {
        const chart = new MultiPlotChart({
            title: "loss",
            titleSpacing: 8,
            titlePosition: LabelPositionFlags.bottom,
            titleForeground: Color.white,
            titleBackground: BackgroundColor.black,
        });

        const cOffset = 1;
        const c1Width = Math.floor(options.width - options.width / 3);
        const c2Width = options.width - c1Width - cOffset;

        const rOffset = 1;
        const r1Height = Math.floor(options.height - options.height / 2);
        const r2Height = options.height - r1Height - rOffset;

        // Loss
        chart.addPlot({
            xOffset: 0, yOffset: 0,
            width: c1Width, height: options.height
        }, {title: "Loss", axisScale: PlotAxisScale.log, aggregation: PlotSeriesAggregationFn.mean, axisLabelsFraction: 4});
        chart.addPlotSeries(0, {color: Color.blue, overflow: PlotSeriesOverflow.logScale});

        // Accuracy
        chart.addPlot({
            xOffset: chart.plots[0].width + 1, yOffset: 0,
            width: c2Width, height: r1Height
        }, {title: "Accuracy", axisLabelsFraction: 4});
        chart.addPlotSeries(1, {color: Color.red, overflow: PlotSeriesOverflow.logScale});

        // Learning rate
        chart.addPlot({
            xOffset: c1Width + cOffset, yOffset: r1Height + rOffset,
            width: c2Width, height: r2Height
        }, {title: "L. Rate", axisScale: PlotAxisScale.logInverted, axisLabelsFraction: 4});
        chart.addPlotSeries(2, {color: Color.yellow, overflow: PlotSeriesOverflow.logScale});

        return chart;
    }
}