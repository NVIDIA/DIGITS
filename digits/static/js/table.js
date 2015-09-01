// Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

(function($, window, document, undefined) {
    var nonNumeric = ['Name', 'ID', 'Status', 'Runtime', 'Framework'];
    var defaultVisible = ['Name', 'Status', 'Runtime', 'Loss', 'Accuracy'];

    function getText(node) {
        return node.innerText || node.textContent;
    }

    var dataTableConfig = {
        dom: 'Brtip',
        order: [1, 'desc'],
        paginate: false,
        searching: true, // column specific search only
        // extensions
        buttons: [
            {   // Metadata (Name, ID, Status, Runtime)
                extend: 'colvis',
                text: 'Metadata',
                columns: ':lt(4)'
            },
            {   // Data from the latest epoch
                extend: 'colvis',
                text: 'Latest',
                columns: function(idx, data, node) {
                    return !getText(node).match(/max$/) &&
                           !getText(node).match(/min$/) &&
                           $.inArray(getText(node), nonNumeric) === -1;
                }
            },
            {   // Max value encountered for each model
                extend: 'colvis',
                text: 'Maximum',
                columns: function(idx, data, node) {
                    return getText(node).match(/max$/);
                }
            },
            {   // Min value encountered for each model
                extend: 'colvis',
                text: 'Minimum',
                columns: function(idx, data, node) {
                    return getText(node).match(/min$/);
                }
            },
            {
                text: 'Show/Hide Filters',
                action: function() {
                    $('.filters').toggleClass('hidden');
                }
            }
        ]
    }

    function getRange(data) {
        if (typeof data !== 'string') return '';

        var matches = data.match(/([^:]*):([^:]*)/);
        var min = Number.NEGATIVE_INFINITY;
        var max = Number.POSITIVE_INFINITY;

        // Simple case--no range searching
        if (matches === null) return data.toString();

        if (matches[1] !== '') min = parseFloat(matches[1]);
        if (matches[2] !== '') max = parseFloat(matches[2]);
        return [min, max];
    }

    function setFooterFilter(table, col) {
        if ($.inArray(getText(col.header()), nonNumeric) !== -1) {
            // Search by regex
            $('input', col.header()).on('keyup change', function() {
                col.search(this.value, true).draw();
            });
        } else {
            // search by range
            $.fn.dataTableExt.afnFiltering.push(function(settings, data, index) {
                var value = data[col.index()];
                var query = $('input', col.header()).val();
                var range = getRange(query);

                if (typeof range === 'string') return value.search(query) !== -1;
                if ($.isArray(range)) {
                    value = parseFloat(value);

                    return range[0] < value && value < range[1];
                }
            });
            $('input', col.header()).on('keyup change', function() {
                table.draw();
            });
        }
    }

    function setDefaultVisibility(col) {
        if ($.inArray(getText(col.header()), defaultVisible) === -1) {
            col.visible(false, false);
        }
    }

    $(document).ready(function() {
        // Prepare footers
        $('#job_table thead th').each(function() {
            var title = $('#job_table thead th').eq($(this).index()).text();
            $(this).append('<input class="form-control filters" type="text" placeholder="'+title+'"/>');
        });

        // Create tables
        var table = $('#job_table').DataTable(dataTableConfig);

        table.columns().every(function() {
            // Set column filters
            setFooterFilter(table, this);
            // Set default visibility.
            setDefaultVisibility(this);
        });

        // Remove form-inline from table so that it will grow/shrink -- this seems
        // to be a bug in dataTables.bootstrap(.min).js.
        $('#job_table_wrapper').removeClass('form-inline');
    });
})($, window, document)
