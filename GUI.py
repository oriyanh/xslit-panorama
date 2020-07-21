import io
import os
import shutil

import PySimpleGUI as sg
from PIL import Image
import engine
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class GUI:

    def __init__(self):
        self.action = None
        self.refocus = None
        self.viewpoint = None
        self.data_path = None
        self.slider_resolution = 3
        self.output_temp = os.path.join(os.getcwd(), 'temp', 'out.png')
        os.makedirs(os.path.dirname(self.output_temp), exist_ok=True)

    def _update_refocus_state(self, values):
        if values[0] != '':
            self.refocus.motion_max = int(values[0])
        if values[1] != '':
            self.refocus.motion_min = int(values[1])
        for i in [6, 5, 4, 3, 2]:
            if values[i]:
                self.slider_resolution = i - 1
                break
        self.refocus._interpolate(float(values['slider']))
        b, g, r = cv.split(self.refocus.output_img)
        im = cv.merge((r, g, b))
        im = cv.resize(im, (400, 250))
        plt.imsave(self.output_temp, im)

    def _welcome_window(self):
        sg.theme('Light Blue 2')  # Add a touch of color
        # All the stuff inside your window.
        layout = [[sg.Text('Path (Video or Photo Album): '), ],
            [sg.Button('Video'), sg.Button('Album')],
            [sg.Checkbox('Right-to-Left?')],
            [sg.Text('Which Kind of Application you want? choose one: ')],
            [sg.Button('Refocusing'), sg.Button('Change Viewpoint')]]

        # Create the Window
        window = sg.Window('Welcome!', layout, element_justification='center', text_justification='center')
        # Event Loop to process "events" and get the "values" of the inputs
        self.src = ''
        while True:
            event, values = window.read()
            # self.is_rtl = values[1]
            self.is_rtl = values[0]
            if event == None:  # if user closes window or clicks cancel
                break
            elif event == 'Refocusing':
                if not os.path.exists(self.src):
                    continue
                self.refocus = engine.Engine()
                self.refocus.refocus(self.src)
                self.refocus._interpolate(150)
                b, g, r = cv.split(self.refocus.output_img)
                im = cv.merge((r, g, b))
                im = cv.resize(im, (400, 250))
                plt.imsave(self.output_temp, im)
                self.action = 'Refocus'
                break
            elif event == 'Change Viewpoint':
                if not os.path.exists(self.src):
                    continue
                self.action = 'Viewpoint'
                self.viewpoint = engine.Engine()
                self.viewpoint.src = self.src
                self.viewpoint.load_images()
                self.viewpoint.init_viewpoint()
                break
            elif event == 'Album':
                self.src = sg.popup_get_folder('Choose album directory')
            elif event == 'Video':
                self.src = sg.popup_get_file('Choose video file',
                                             file_types=(("VIDEOS", ".mp4"),("VIDEOS", ".mov"),
                                                         ("VIDEOS", ".flv"),("VIDEOS", ".mpeg") ,
                                                         ("VIDEOS", ".mpg"),("VIDEOS", ".avi")))
        window.close()

    def _refocus_window(self):
        sg.theme('Light Blue 2')  # Add a touch of color
        # All the stuff inside your window.
        col = [[sg.Text('Object Motion ')],
               [sg.Text('Nearest Object: '), sg.InputText()],
               [sg.Text('Farthest Object: '), sg.InputText()],
               [sg.Text('Sensitivity: ')],
               [sg.CB('Highest'), sg.CB('High'), sg.CB('Normal'), sg.CB('Low'), sg.CB('Lowest')],
               [sg.Button('Compute'), sg.Button('Save')]]

        layout = [[sg.Text('        Refocusing ')],
                  [sg.Image(self.output_temp, key='image')],
                  [sg.Slider(range=(self.refocus.motion_min, self.refocus.motion_max), resolution=self.slider_resolution,  default_value=self.refocus.motion, orientation='v', size=(8, 20), key='slider'), sg.Column(col)]]
        # Create the Window
        window = sg.Window('Refocus', layout)
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()
            if event == None:  # if user closes window or clicks cancel
                break
            elif event == 'Compute':
                self._update_refocus_state(values)
                window.close()
                self._refocus_window()
            elif event == 'Save':
                dst = sg.popup_get_file("Save output image to disk", title='Save Image',
                                        save_as=True, default_extension=".png",
                                        file_types=(("IMAGES", ".png"),("IMAGES", ".jpg")))
                b, g, r = cv.split(self.refocus.output_img*255)
                img = Image.fromarray(cv.merge((r, g,b)).astype(np.uint8))
                img.save(dst)
        window.close()

    def _update_viewpoint(self, frame1, col1, frame2, col2, angle=None):
        self.viewpoint.change_viewpoint(self.src, frame1, col1, frame2, col2, angle=angle, is_rtl=self.is_rtl)
        w, h, _ = self.viewpoint.output_img.shape
        b, g, r = cv.split(self.viewpoint.output_img)
        w_out = 800
        h_out = int(w*h/w_out)
        img = Image.fromarray(np.dstack((r, g, b)), mode='RGB')
        img.resize((w_out, h_out))
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        del img
        return bio.getvalue()

    def _viewpoint_display_window(self):
        sg.theme('Light Blue 2')  # Add a touch of color
        # All the stuff inside your window.
        w, h, _ = self.viewpoint.output_img.shape
        b, g, r = cv.split(self.viewpoint.output_img)
        w_out = 800
        h_out = int(w*h/w_out)
        img = Image.fromarray(np.dstack((r, g, b)), mode='RGB')
        img.resize((h_out, w_out))
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        del img
        layout = [[sg.Text('Change Viewpoint ')],
                  [sg.Button('Zoom In')],
                  [sg.Button('Previous'), sg.Image(data=bio.getvalue(), key='image'),
                   sg.Button('Next')],
                  [sg.Button('Viewpoint Left'), sg.Button('Zoom Out'),
                   sg.Button('Viewpoint Right')],
                  [sg.Text('First Frame:'), sg.InputText(size=(4, 4), default_text=f'{self.viewpoint.frame1+1}', key='frame1'),
                   sg.Text('Last Frame:'), sg.InputText(size=(4, 4), default_text=f'{self.viewpoint.frame2}', key='frame2'),
                   sg.Text(f'(between 1 and {self.viewpoint.num_frames}, inclusive)')],
                   [sg.Text('First column:'),
                   sg.InputText(size=(4, 4), default_text=f'{self.viewpoint.col1+1}', key='col1'),
                   sg.Text('Last column:'), sg.InputText(size=(4, 4), default_text=f'{self.viewpoint.col2}', key='col2'),
                    sg.Text(f'(between 1 and {self.viewpoint.width}, inclusive)')],
                  [sg.Button('Rotate Time Slice')],
                  [sg.Button('Save'), sg.Button('Update')]]
        # Create the Window
        window = sg.Window('Change Viewpoint', layout, element_justification='center', text_justification='center')
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()

            if event == None:  # if user closes window or clicks cancel
                break
            elif event == 'Previous':
                if self.viewpoint.frame1 == 0:
                    print("No previous frame")
                else:
                    output = self._update_viewpoint(self.viewpoint.frame1 - 1, self.viewpoint.col1,
                                           self.viewpoint.frame2 - 1, self.viewpoint.col2)
                    window['image'].Update(data=output)
                    window['frame1'].Update(f'{self.viewpoint.frame1}')
                    window['frame2'].Update(f'{self.viewpoint.frame2}')
                    window['col1'].Update(f'{self.viewpoint.col1}')
                    window['col2'].Update(f'{self.viewpoint.col2}')
            elif event == 'Next':
                num_frames = self.viewpoint.num_frames
                if self.viewpoint.frame2 == num_frames:
                    print("No next frame")
                else:
                    output = self._update_viewpoint(self.viewpoint.frame1 + 1, self.viewpoint.col1,
                                           self.viewpoint.frame2 + 1, self.viewpoint.col2)
                    window['image'].Update(data=output)
                    window['frame1'].Update(f'{self.viewpoint.frame1}')
                    window['frame2'].Update(f'{self.viewpoint.frame2}')
                    window['col1'].Update(f'{self.viewpoint.col1}')
                    window['col2'].Update(f'{self.viewpoint.col2}')
            elif event == 'Viewpoint Right':
                if self.viewpoint.col1 == 0:
                    print("can't move right")
                else:
                    output = self._update_viewpoint(self.viewpoint.frame1, self.viewpoint.col1 - 10,
                                           self.viewpoint.frame2, self.viewpoint.col2 - 10)
                    window['image'].Update(data=output)
                    window['frame1'].Update(f'{self.viewpoint.frame1}')
                    window['frame2'].Update(f'{self.viewpoint.frame2}')
                    window['col1'].Update(f'{self.viewpoint.col1}')
                    window['col2'].Update(f'{self.viewpoint.col2}')
            elif event == 'Viewpoint Left':
                width = self.viewpoint.width
                if self.viewpoint.col2 == width:
                    print("can't move left")
                else:
                    output = self._update_viewpoint(self.viewpoint.frame1, self.viewpoint.col1 + 10,
                                           self.viewpoint.frame2, self.viewpoint.col2 + 10)
                    window['image'].Update(data=output)
                    window['frame1'].Update(f'{self.viewpoint.frame1}')
                    window['frame2'].Update(f'{self.viewpoint.frame2}')
                    window['col1'].Update(f'{self.viewpoint.col1}')
                    window['col2'].Update(f'{self.viewpoint.col2}')

            elif event == 'Zoom In':
                diff = 10
                col1 = self.viewpoint.col1 + diff
                col2 = self.viewpoint.col2 - diff

                if col1 > self.viewpoint.width:
                    col1 = self.viewpoint.width
                elif col1 == self.viewpoint.width:
                    print("Can't zoom in anymore")
                    continue
                if col2 < 0:
                    col2 = 0
                elif col2 == 0:
                    print("Can't zoom in anymore")
                    continue

                output = self._update_viewpoint(self.viewpoint.frame1, col1,
                                       self.viewpoint.frame2, col2)
                window['image'].Update(data=output)
                window['frame1'].Update(f'{self.viewpoint.frame1}')
                window['frame2'].Update(f'{self.viewpoint.frame2}')
                window['col1'].Update(f'{self.viewpoint.col1}')
                window['col2'].Update(f'{self.viewpoint.col2}')
            elif event == 'Zoom Out':
                diff = 10
                col1 = self.viewpoint.col1 - diff
                col2 = self.viewpoint.col2 + diff
                if col2 > self.viewpoint.width:
                    col2 = self.viewpoint.width
                if col1 < 0:
                    col1 = 0
                if self.viewpoint.col2 != col2 or self.viewpoint.col1 != col1:
                    output = self._update_viewpoint(self.viewpoint.frame1, col1,
                                                    self.viewpoint.frame2, col2)
                    window['image'].Update(data=output)
                    window['frame1'].Update(f'{self.viewpoint.frame1}')
                    window['frame2'].Update(f'{self.viewpoint.frame2}')
                    window['col1'].Update(f'{self.viewpoint.col1}')
                    window['col2'].Update(f'{self.viewpoint.col2}')
                else:
                    print("Can't zoom out anymore")
            elif event == 'Update':
                frame1 = int(values['frame1'])
                frame2 = int(values['frame2'])
                col1 = int(values['col1'])
                col2 = int(values['col2'])
                output = self._update_viewpoint(frame1, col1,
                                       frame2, col2)
                window['image'].Update(data=output)
            elif event == 'Save':
                dst = sg.popup_get_file("Save output image to disk",
                                        title='Save Image', save_as=True,
                                        default_extension=".png",
                                        file_types=(("IMAGES", ".png"), ("IMAGES", ".jpg")))
                if dst:
                    b, g, r = cv.split(self.viewpoint.output_img)
                    img = Image.fromarray(cv.merge((r, g, b)))
                    img.save(dst)
            elif event == 'Rotate Time Slice':
                val = sg.popup_get_text(f"Set time slice to N degrees, where N is an number in range [0,360)\n (0 is pushbroom panorama)", title="Rotate time slice angle")
                if val is None:
                    continue
                angle = float(val)
                frame1 = int(values['frame1'])
                frame2 = int(values['frame2'])
                col1 = int(values['col1'])
                col2 = int(values['col2'])
                output = self._update_viewpoint(frame1, col1,
                                                frame2, col2, angle)
                window['image'].Update(data=output)
                window['frame1'].Update(f'{self.viewpoint.frame1}')
                window['frame2'].Update(f'{self.viewpoint.frame2}')
                window['col1'].Update(f'{self.viewpoint.col1}')
                window['col2'].Update(f'{self.viewpoint.col2}')
        window.close()

    def _viewpoint_selection_window(self):
        sg.theme('Light Blue 2')  # Add a touch of color
        # All the stuff inside your window.
        self.viewpoint.src = self.src
        self.viewpoint.load_images()
        layout = [[sg.Text('        Change Viewpoint ')],
                  [sg.Text('First Frame:'), sg.InputText(size=(4, 4), default_text='1'), sg.Text(f'(Of {self.viewpoint.num_frames} Frames, inclusive)'),
                   sg.Text('Last Frame:'), sg.InputText(size=(4, 4), default_text=str(self.viewpoint.num_frames)),
                   sg.Text(f'(Of {self.viewpoint.num_frames} Frames)')],
                  [sg.Text('First Column:'), sg.InputText(size=(4, 4), default_text='1'), sg.Text(f'(Of {self.viewpoint.width} columns, inclusive)'),
                   sg.Text('Last Column:'), sg.InputText(size=(4, 4), default_text=str(self.viewpoint.width)),
                   sg.Text(f'(Of {self.viewpoint.width} columns)')],
                  [sg.Button('Compute')]]
        # Create the Window
        window = sg.Window('Change Viewpoint', layout, text_justification='center', element_justification='center')
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()
            frame1 = int(values[0]) - 1
            frame2 = int(values[1])
            col1 = int(values[2]) - 1
            col2 = int(values[3])
            if event == None:  # if user closes window or clicks cancel
                break
            elif event == 'Compute':
                self._update_viewpoint(frame1, col1, frame2, col2)
                break
        window.close()

    def loop(self):

        self._welcome_window()
        if self.action == 'Refocus':
            self._refocus_window()
        elif self.action == 'Viewpoint':
            self._viewpoint_selection_window()
            self._viewpoint_display_window()

    def __del__(self):
        shutil.rmtree(os.path.dirname(self.output_temp), ignore_errors=True)
