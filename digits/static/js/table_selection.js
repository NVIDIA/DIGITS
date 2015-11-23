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
    this.mousedown_last_rowIndex = -1;

    function on_selection_changed() {
        var e = $.Event('selection_changed');
        e.count = $(TableSelection.current_table + ' tr.selected').length;
        $(window).trigger(e);
    }

    // Disable text selection because we want to select rows not text
    $(table_id + ' tbody').disableSelection();

    // track the current table for key events
    $(table_id + ' tbody').mouseenter(function() {
        TableSelection.current_table = table_id;
    });

    // track the current table for key events
    $(table_id + ' tbody').mouseleave(function() {
        TableSelection.current_table = null;
    });

    // call this that for use in anonymous functions
    var that = this;

    // add 'selected' class to item
    function select(item) {
        if ($(item).not('.selected').length > 0) {
            $(item).addClass('selected');
            on_selection_changed()
        }
    }

    // remote 'selected' class from item
    function deselect(item) {
        if ($(item).hasClass('selected')) {
            $(item).removeClass('selected');
            on_selection_changed();
        }
    }

    // begin a drag selection
    $(table_id + ' tbody').on('mousedown', 'tr', function(e) {
        that.mousedown_row = this;
        that.mousedown_with_alt = (e.ctrlKey || e.metaKey);
        that.mousedown_with_shift = (e.shiftKey);
        that.mousedown_was_selected = $(this).hasClass('selected');
        that.mousedown_last_rowIndex = -1;
    });

    // drag selection
    $(table_id + ' tbody').on('mousemove', 'tr', function(e) {
        if (e.which == 1 && that.mousedown_row != null) {

            if (that.mousedown_last_rowIndex == this.rowIndex)
                return;

            var first = Math.min(that.mousedown_row.rowIndex, this.rowIndex) + 1;
            var last = Math.max(that.mousedown_row.rowIndex, this.rowIndex) + 1;

            // Do we want to add, remove, or replace the selection
            var action = (that.mousedown_with_shift ? 'add' :
                          that.mousedown_with_alt ? (
                              that.mousedown_was_selected ? 'remove' : 'add') :
                          'replace');

            if (action == 'remove') {
                for (var i = first; i <= last; i++)
                    deselect($(table_id + ' tr').eq(i));
                that.last_selected_row = $(TableSelection.current_table + ' tr.selected').last();
            } else {
                if (action == 'replace')
                    deselect($(table_id + ' tr'))

                for (var i = first; i <= last; i++)
                    select($(table_id + ' tr').eq(i));

                that.last_selected_row = this;
            }
            that.mousedown_last_rowIndex = this.rowIndex;
        }
    });

    // end drag selection
    $(table_id + ' tbody').on('mouseup', 'tr', function() {
        that.mousedown_row = null;
    });

    // Let links continue to work.  Without this, a click would
    // only do a row selection.
    $(table_id + ' tbody').on('click', 'a', function(e) {
        location.href = $(this).attr('href');
    });

    // click selection
    $(table_id + ' tbody').on('click', 'tr', function(e) {
        e.preventDefault();
        // shift select region
        if ((e.shiftKey)) {
            if (that.last_selected_row != null) {
                var first = Math.min(that.last_selected_row.rowIndex, this.rowIndex) + 1;
                var last = Math.max(that.last_selected_row.rowIndex, this.rowIndex) + 1;
                for (var i = first; i <= last; i++) {
                    select($(table_id + ' tr').eq(i));
                }
                that.last_selected_row = this;
            }
        }
        // toggle click selection
        else if ((e.ctrlKey || e.metaKey)) {
            if ($(this).hasClass('selected')) {
                deselect($(this));
                that.last_selected_row = $(TableSelection.current_table + ' tr.selected').last();
            } else {
                select($(this));
                that.last_selected_row = this;
            }
        }
        // simple unmodified click selection
        else {
            deselect($(table_id + ' tr'));
            select($(this));
            that.last_selected_row = this;
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
                that.last_selected_row = null;
            }
            // Use alt-d for deselect all rows
            if ((e.ctrlKey || e.metaKey) && e.keyCode == 68) {
                e.preventDefault();
                deselect($(TableSelection.current_table + ' tr'));
                that.last_selected_row = null;
            }
            // Use alt-i for invert selection
            if ((e.ctrlKey || e.metaKey) && e.keyCode == 73) {
                e.preventDefault();
                $(TableSelection.current_table + ' tr.selected').addClass('swap');
                $(TableSelection.current_table + ' tr').not('.swap').addClass('selected');
                $(TableSelection.current_table + ' tr.swap').removeClass('swap selected');
                that.last_selected_row = $(TableSelection.current_table + ' tr.selected').last();

                // just trigger the selection_changed event directly
                on_selection_changed();
            }
            switch(e.which) {
            case 38: // up
                if (that.last_selected_row) {
                    if (!e.shiftKey)
                        deselect($(TableSelection.current_table + ' tr'));
                    var row = $(that.last_selected_row).prev('tr');
                    if (row.length)
                        that.last_selected_row = row;
                    select($(that.last_selected_row));
                }
                break;
            case 40: // down
                if (that.last_selected_row != null) {
                    if (!e.shiftKey)
                        deselect($(TableSelection.current_table + ' tr'));
                    var row = $(that.last_selected_row).next('tr');
                    if (row.length)
                        that.last_selected_row = row;
                    select($(that.last_selected_row));
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
