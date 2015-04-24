// Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
function drawCombinedGraph(data) {
    $(".combined-graph").show();
    c3.generate($.extend({
        bindto: '#combined-graph',
        axis: {
            x: {
                label: {
                    text: 'Epoch',
                    position: 'outer-center',
                },
                tick: {
                    // 3 sig-digs
                    format: function(x) { return Math.round(x*1000)/1000; },
                    fit: false,
                },
                min: 0,
                padding: {left: 0},
            },
            y: {
                label: {
                    text: 'Loss',
                    position: 'outer-middle',
                },
                min: 0,
                padding: {bottom: 0},
            },
            y2: {
                show: true,
                label: {
                    text: 'Accuracy (%)',
                    position: 'outer-middle',
                },
                min: 0,
                max: 100,
                padding: {top: 0, bottom: 0},
            },
        },
        grid: {x: {show: true} },
        legend: {position: 'bottom'},
    },
    {data: data}
    ));
}
function drawLRGraph(data) {
    $(".lr-graph").show();
    c3.generate($.extend({
        bindto: '#lr-graph',
        size: {height: 300},
        axis: {
            x: {
                label: {
                    text: 'Epoch',
                    position: 'outer-center',
                },
                tick: {
                    // 3 sig-digs
                    format: function(x) { return Math.round(x*1000)/1000; },
                    fit: false,
                },
                min: 0,
                padding: {left: 0},
            },
            y: {
                label: {
                    text: 'Learning Rate',
                    position: 'outer-middle',
                },
                min: 0,
                padding: {bottom: 0},
            },
        },
        grid: {x: {show: true} },
        legend: {show: false},
    },
    {data: data}
    ));
}
