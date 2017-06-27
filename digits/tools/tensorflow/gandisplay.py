import time
import numpy as np
import wx

# This has been set up to optionally use the wx.BufferedDC if
# USE_BUFFERED_DC is True, it will be used. Otherwise, it uses the raw
# wx.Memory DC , etc.

# USE_BUFFERED_DC = False
USE_BUFFERED_DC = True

myEVT = wx.NewEventType()
DISPLAY_GRID_EVT = wx.PyEventBinder(myEVT, 1)


class MyEvent(wx.PyCommandEvent):
    """Event to signal that a count value is ready"""
    def __init__(self, etype, eid, value=None):
        """Creates the event object"""
        wx.PyCommandEvent.__init__(self, etype, eid)
        self._value = value

    def GetValue(self):
        """Returns the value from the event.
        @return: the value of this event

        """
        return self._value


class BufferedWindow(wx.Window):

    """

    A Buffered window class.

    To use it, subclass it and define a Draw(DC) method that takes a DC
    to draw to. In that method, put the code needed to draw the picture
    you want. The window will automatically be double buffered, and the
    screen will be automatically updated when a Paint event is received.

    When the drawing needs to change, you app needs to call the
    UpdateDrawing() method. Since the drawing is stored in a bitmap, you
    can also save the drawing to file by calling the
    SaveToFile(self, file_name, file_type) method.

    """
    def __init__(self, *args, **kwargs):
        # make sure the NO_FULL_REPAINT_ON_RESIZE style flag is set.
        kwargs['style'] = kwargs.setdefault('style', wx.NO_FULL_REPAINT_ON_RESIZE) | wx.NO_FULL_REPAINT_ON_RESIZE
        wx.Window.__init__(self, *args, **kwargs)

        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)

        # OnSize called to make sure the buffer is initialized.
        # This might result in OnSize getting called twice on some
        # platforms at initialization, but little harm done.
        self.OnSize(None)
        self.paint_count = 0

    def Draw(self, dc):
        # just here as a place holder.
        # This method should be over-ridden when subclassed
        pass

    def OnPaint(self, event):
        # All that is needed here is to draw the buffer to screen
        if USE_BUFFERED_DC:
            dc = wx.BufferedPaintDC(self, self._Buffer)
        else:
            dc = wx.PaintDC(self)
            dc.DrawBitmap(self._Buffer, 0, 0)

    def OnSize(self, event):
        # The Buffer init is done here, to make sure the buffer is always
        # the same size as the Window
        # Size = self.GetClientSizeTuple()
        Size = self.ClientSize

        # Make new offscreen bitmap: this bitmap will always have the
        # current drawing in it, so it can be used to save the image to
        # a file, or whatever.
        self._Buffer = wx.EmptyBitmap(*Size)
        self.UpdateDrawing()

    def SaveToFile(self, FileName, FileType=wx.BITMAP_TYPE_PNG):
        # This will save the contents of the buffer
        # to the specified file. See the wxWindows docs for
        # wx.Bitmap::SaveFile for the details
        self._Buffer.SaveFile(FileName, FileType)

    def UpdateDrawing(self):
        """
        This would get called if the drawing needed to change, for whatever reason.

        The idea here is that the drawing is based on some data generated
        elsewhere in the system. If that data changes, the drawing needs to
        be updated.

        This code re-draws the buffer, then calls Update, which forces a paint event.
        """
        dc = wx.MemoryDC()
        dc.SelectObject(self._Buffer)
        self.Draw(dc)
        del dc  # need to get rid of the MemoryDC before Update() is called.
        self.Refresh()
        self.Update()


class DrawWindow(BufferedWindow):
    def __init__(self, *args, **kwargs):
        # Any data the Draw() function needs must be initialized before
        # calling BufferedWindow.__init__, as it will call the Draw function.
        self.DrawData = {}
        BufferedWindow.__init__(self, *args, **kwargs)

    def Draw(self, dc):
        dc.SetBackground(wx.Brush("White"))
        dc.Clear()  # make sure you clear the bitmap!

        # Here's the actual drawing code.
        for key, data in self.DrawData.items():
            if key == "text":
                dc.DrawText(data, 0, 0)
            elif key == "np":
                data = data.astype('uint8')
                img_count = data.shape[0]
                height = data.shape[1]
                width = data.shape[2]

                grid_size = int(np.sqrt(img_count))

                size = (grid_size * width, grid_size * height)

                if True:  # self.size != size:
                    self.size = size
                    self.SetSize(size)

                image = wx.EmptyImage(width, height)

                for i in xrange(img_count):
                    x = width * (i // grid_size)
                    y = height * (i % grid_size)
                    s = data[i].tostring()
                    image.SetData(s)

                    wxBitmap = image.ConvertToBitmap()
                    dc.DrawBitmap(wxBitmap, x=x, y=y)


class TestFrame(wx.Frame):

    SLIDER_WIDTH = 100
    SLIDER_BORDER = 50
    STATUS_HEIGHT = 20

    def __init__(self, parent=None, grid_size=640, attributes=[]):
        wx.Frame.__init__(self, parent,
                          size=(grid_size + self.SLIDER_WIDTH + self.SLIDER_BORDER, grid_size + self.STATUS_HEIGHT),
                          title="GAN Demo",
                          style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)

        # Set up the MenuBar
        MenuBar = wx.MenuBar()

        file_menu = wx.Menu()

        item = file_menu.Append(wx.ID_EXIT, text="&Exit")
        self.Bind(wx.EVT_MENU, self.OnQuit, item)
        MenuBar.Append(file_menu, "&File")

        self.SetMenuBar(MenuBar)

        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText('Initialising...')

        # Set up UI elements
        panel = wx.Panel(self)
        self.Window = DrawWindow(panel, size=(grid_size, grid_size))

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.Window, 1, wx.ALIGN_LEFT)

        # Sliders
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.speed_slider = wx.Slider(panel, -1, value=5, minValue=0, maxValue=10, pos=wx.DefaultPosition,
                                      size=(self.SLIDER_WIDTH, -1),
                                      style=wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)

        slider_text = wx.StaticText(panel, label='Speed')
        vbox.Add(slider_text, 0, wx.ALIGN_CENTRE)
        vbox.Add(self.speed_slider, 0, wx.ALIGN_CENTRE)

        self.attribute_sliders = []
        for attribute in attributes:
            slider_text = wx.StaticText(panel, label=attribute)
            slider = wx.Slider(panel, -1, value=0, minValue=-100, maxValue=100, pos=wx.DefaultPosition,
                               size=(self.SLIDER_WIDTH, -1),
                               style=wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)

            vbox.Add(slider_text, 0, wx.ALIGN_CENTRE)
            vbox.Add(slider, 0, wx.ALIGN_CENTRE)
            self.attribute_sliders.append(slider)

        hbox.Add(vbox, 0, wx.ALIGN_RIGHT)
        panel.SetSizer(hbox)

        self.Window.DrawData = {'text': u'Initialising...'}
        self.Window.UpdateDrawing()

        # to measure frames per second
        self.last_frame_timestamp = None
        self.last_fps_update = None

        # add panel to frame
        frameSizer = wx.BoxSizer(wx.VERTICAL)
        frameSizer.Add(panel, 0, wx.EXPAND | wx.ALIGN_LEFT)
        self.SetSizer(frameSizer)

        self.Show()

        self.Fit()

        self.Bind(DISPLAY_GRID_EVT, self.OnDisplayCell)

    def OnQuit(self, event):
        self.Close(True)

    def OnDisplayCell(self, evt):
        array = evt.GetValue()
        self.Window.DrawData = {'np': array}
        self.Window.UpdateDrawing()

        if self.last_frame_timestamp is not None:
            fps = 1. / (time.time() - self.last_frame_timestamp)
            if (self.last_fps_update is None) or (time.time() - self.last_fps_update > 0.5):
                self.statusbar.SetStatusText('%.1ffps' % fps)
                self.last_fps_update = time.time()
        self.last_frame_timestamp = time.time()


class DemoApp(wx.App):

    def __init__(self, arg, grid_size, attributes):
        self.gan_grid_size = grid_size
        self.attributes = attributes
        super(DemoApp, self).__init__(arg)

    def OnInit(self):
        self.frame = TestFrame(grid_size=self.gan_grid_size, attributes=self.attributes)
        self.SetTopWindow(self.frame)
        return True

    def DisplayCell(self, array):
        evt = MyEvent(myEVT, -1, array)
        wx.PostEvent(self.frame, evt)

    def GetSpeed(self):
        return self.frame.speed_slider.GetValue()

    def GetAttributes(self):
        return [s.GetValue() for s in self.frame.attribute_sliders]
