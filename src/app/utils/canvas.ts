export function drawNeuron(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number, width: number, color: string) {
    ctx.strokeStyle = "solid";
    ctx.lineWidth = width;
    ctx.fillStyle = color;

    ctx.beginPath();
    ctx.arc(x - radius / 2, y - radius / 2, radius, 0, 2 * Math.PI, false);
    ctx.closePath();

    ctx.fill();
    ctx.stroke();
}