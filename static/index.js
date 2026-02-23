document.addEventListener('DOMContentLoaded', function() {
    var canvas = document.getElementById("canvas");
    var context = canvas.getContext("2d");
    canvas.width = 664;
    canvas.height = 373;
    var Mouse = { x: 0, y: 0 };
    var lastMouse = { x: 0, y: 0 };
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "black";
    context.lineWidth = 10;
    context.lineJoin = 'round';
    context.lineCap = 'round';

    canvas.style.cursor = "crosshair";

    canvas.addEventListener("mousemove", function(e) {
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;
        Mouse.x = e.pageX - this.offsetLeft;
        Mouse.y = e.pageY - this.offsetTop;
    }, false);

    canvas.addEventListener("mousedown", function(e) {
        canvas.addEventListener("mousemove", onPaint, false);
    }, false);

    canvas.addEventListener("mouseup", function() {
        canvas.removeEventListener("mousemove", onPaint, false);
    }, false);

    var onPaint = function() {
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
        context.closePath();
        context.stroke();
    };

    var clearButton = document.getElementById("clearButton");
    clearButton.addEventListener("click", function() {
        context.clearRect(0, 0, 664, 373);
        context.fillStyle = "white";
        context.fillRect(0, 0, canvas.width, canvas.height);
    });

    var slider = document.getElementById("myRange");
    var output = document.getElementById("sliderValue");
    output.innerHTML = slider.value;
    slider.oninput = function() {
        output.innerHTML = this.value;
        context.lineWidth = this.value;
    };
});
