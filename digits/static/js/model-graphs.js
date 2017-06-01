// Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
function drawCombinedGraph(data) {
    $('.combined-graph').show();
    // drawCombinedGraph.chart is a static variable that holds the graph state;
    // it is initialized on first call to drawCombinedGraph()
    if (typeof drawCombinedGraph.chart == 'undefined') {
        // create instance of C3 chart
        drawCombinedGraph.chart = c3.generate($.extend({
            bindto: '#combined-graph',
            axis: {
                x: {
                    label: {
                        text: 'Epoch',
                        position: 'outer-center',
                    },
                    tick: {
                        // 3 sig-digs
                        format: function(x) { return Math.round(x * 1000) / 1000; },
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
        {
            data: data,
            transition: {
                duration: 0,
            },
            subchart: {
                show: true,
            },
            zoom: {
                rescale: true,
            },
        }
        ));
    }
    else
    {
        // just update data
        drawCombinedGraph.chart.load(data);
        drawCombinedGraph.chart.data.names(data.names);
    }
}
function drawLRGraph(data) {
    $('.lr-graph').show();
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
                    format: function(x) { return Math.round(x * 1000) / 1000; },
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
