// Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

// class to setup the event handling to add extended selection for a
// dataTable. This allows multiple dataTables to function on the same
// page. This adds mouse/pointer based click and marquee selection
// with shift and ctrl/meta modification for range and toggle
// selection, up/down arrow key also with shift and ctrl/meta
// modification, and alt-a, alt-d, and alt-i for select-all,
// deselect-all, and invert-selection.

// Because key events seem to need to go through the document, the
// current table is tracked so that the correct table can be
// affected. There's a chance that would fail if someone could put a
// subdataTable inside of another dataTable, but that's unlikely if
// not impossible.

// var table = $('#my_table').DataTable();
// var ts = new TableSelection('#my_table');

function TableSelection(table_id) {
    // class state variables
    this.last_selected_row = null;
    this.mousedown_row = null;
    this.mousedown_with_alt = false;
    this.mousedown_with_shift = false;
    this.mousedown_was_selected = false;
    this.selected = [];
    ts = this;
    function on_selection_changed() {
        var e = $.Event('selection_changed');
        ts.selected = $(TableSelection.current_table + ' tr.selected');
        e.count = ts.selected.length;
        $(window).trigger(e);
    }

    // Disable text selection because we want to select rows not text
    $(table_id).addClass('selectable');
    // $(table_id).disableSelection();

    // track the current table for key events
    // should be $(table_id + ' tbody').mouseenter(function() {
    $(table_id).mouseenter(function() {
        TableSelection.current_table = table_id;
        TableSelection.current_ts = this;
    });

    // track the current table for key events
    // should be $(table_id).mouseleave(function() {
    $(table_id).mouseleave(function() {
        TableSelection.current_ts.mousedown_row = null;
        TableSelection.current_table = null;
        TableSelection.current_ts = null;
    });

    // add 'selected' class to item
    function select(item) {
        var list = $(item).not(":hidden").not('.selected').not('.unselectable');
        if (list.length > 0) {
            list.addClass('selected');
            on_selection_changed();

            // $('html, body').animate({
            //    scrollTop: $(item).offset().top
            // }, 1000);
        }
    }

    // remote 'selected' class from item
    function deselect(item) {
        if ($(item).hasClass('selected')) {
            $(item).removeClass('selected');
            on_selection_changed();
        }
    }

    function do_selection(tr) {
        if ($(tr).hasClass('unselectable')) {
            return;
        }

        var first = tr.rowIndex;
        var last = tr.rowIndex;

        if (TableSelection.current_ts.mousedown_with_shift &&
            TableSelection.current_ts.last_selected_row) {
            first = Math.min(TableSelection.current_ts.last_selected_row.rowIndex, tr.rowIndex);
            last = Math.max(TableSelection.current_ts.last_selected_row.rowIndex, tr.rowIndex);
        } else if (TableSelection.current_ts.mousedown_row) {
            first = Math.min(TableSelection.current_ts.mousedown_row.rowIndex, tr.rowIndex);
            last = Math.max(TableSelection.current_ts.mousedown_row.rowIndex, tr.rowIndex);
        }

        // Do we want to add, remove, or replace the selection
        var action = (TableSelection.current_ts.mousedown_with_shift ? 'add' :
                      TableSelection.current_ts.mousedown_with_alt ? (
                          TableSelection.current_ts.mousedown_was_selected ? 'remove' : 'add') :
                      'replace');

        if (action == 'remove') {
            for (var i = first; i <= last; i++)
                deselect($(table_id + ' tr').eq(i));
            TableSelection.current_ts.last_selected_row = $(TableSelection.current_table + ' tr.selected').last();
        } else {
            if (action == 'replace')
                deselect($(table_id + ' tr'))

            for (var i = first; i <= last; i++)
                select($(table_id + ' tr').eq(i));

            TableSelection.current_ts.last_selected_row = tr;
        }
    }

    // begin a drag selection
    $(table_id).on('mousedown', 'tr', function(e) {
        if (e.which != 1) {
            return;
        }
        if ($(this).hasClass('unselectable')) {
            return;
        }
        TableSelection.current_ts.mousedown_with_alt = (e.ctrlKey || e.metaKey);
        TableSelection.current_ts.mousedown_with_shift = (e.shiftKey);
        TableSelection.current_ts.mousedown_was_selected = $(this).hasClass('selected');
        TableSelection.current_ts.mousedown_row = null;
        do_selection(this);
        TableSelection.current_ts.mousedown_row = this;
    });

    // drag selection
    $(table_id).on('mousemove', 'tr', function(e) {
        if (e.which != 1 || TableSelection.current_ts.mousedown_row == null) {
            return;
        }
        if (TableSelection.current_ts.last_selected_row == this) {
            return;
        }
        if ($(this).hasClass('unselectable')) {
            return;
        }

        do_selection(this);
    });

    // end drag selection
    $(table_id).on('mouseup', 'tr', function(e) {
        if (e.which != 1 || TableSelection.current_ts.mousedown_row == null) {
            return;
        }
        TableSelection.current_ts.mousedown_row = null;
    });

    // Add these three events to prevent row selection when clicking a link.
    $(table_id).on('mousedown', 'a', function(e) {
        e.stopPropagation();
    });

    $(table_id).on('mousedrag', 'a', function(e) {
        e.stopPropagation();
    });

    $(table_id).on('mouseup', 'a', function(e) {
        e.stopPropagation();
    });

    // deselect when the mouse is clicked outside the table
    $(document).on('click', 'body', function(e) {
        var tag = e.target.tagName.toLowerCase();
        if (tag == 'input' || tag == 'textarea' || tag == 'button' || tag == 'a') {
            return;
        }

        if (TableSelection.current_table == null) {
            deselect($('tr'));
        }
    });

    // Because the key event is global, only register once.
    if (!TableSelection.initialized) {
        TableSelection.initialized = true;
        $(document).keydown(function(e) {
            if (TableSelection.current_table == null) return;
            // Use alt-a for select all rows
            if ((e.ctrlKey || e.metaKey) && e.keyCode == 65) {
                e.preventDefault();
                select($(TableSelection.current_table + ' tr'));
                TableSelection.current_ts.last_selected_row = null;
            }
            // Use alt-d for deselect all rows
            if ((e.ctrlKey || e.metaKey) && e.keyCode == 68) {
                e.preventDefault();
                deselect($(TableSelection.current_table + ' tr'));
                TableSelection.current_ts.last_selected_row = null;
            }
            // Use alt-i for invert selection
            if ((e.ctrlKey || e.metaKey) && e.keyCode == 73) {
                e.preventDefault();
                $(TableSelection.current_table + ' tr.selected').addClass('swap');
                select($(TableSelection.current_table + ' tr').not('.swap'));
                $(TableSelection.current_table + ' tr.swap').removeClass('swap selected');
                TableSelection.current_ts.last_selected_row = $(TableSelection.current_table + ' tr.selected').last();

                // just trigger the selection_changed event directly
                on_selection_changed();
            }
            switch(e.which) {
            case 38: // up
                if (TableSelection.current_ts.last_selected_row) {
                    if (!e.shiftKey)
                        deselect($(TableSelection.current_table + ' tr'));
                    var row = $(TableSelection.current_ts.last_selected_row).prev('tr');
                    while (row.length && ($(row).hasClass('unselectable') || $(row).is(':hidden')))
                        row = $(row).prev('tr');
                    if (row.length)
                        TableSelection.current_ts.last_selected_row = row[0];
                    select($(TableSelection.current_ts.last_selected_row));
                }
                break;
            case 40: // down
                if (TableSelection.current_ts.last_selected_row != null) {
                    if (!e.shiftKey)
                        deselect($(TableSelection.current_table + ' tr'));
                    var row = $(TableSelection.current_ts.last_selected_row).next('tr');
                    while (row.length && ($(row).hasClass('unselectable') || $(row).is(':hidden')))
                        row = $(row).next('tr');
                    if (row.length)
                        TableSelection.current_ts.last_selected_row = row[0];
                    select($(TableSelection.current_ts.last_selected_row));
                }
                break;
            default: return; // exit this handler for other keys
            }
            e.preventDefault(); // prevent the default action (scroll / move caret)
        });
    }
}

// static class variables
TableSelection.current_table = null;
TableSelection.initialized = false;
