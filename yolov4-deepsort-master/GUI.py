import os
import PySimpleGUI as sg
from object_tracker import find
import random

layout = [
    [sg.Text("Ścieżka do pliku wejściowego:", size=(30,1)),
        sg.In(size=(50, 1), enable_events=True, key="-VIDEOIN-"),
        sg.FileBrowse()],
    [sg.Text("Ścieżka do wynikowego pliku video: ", size=(30,1)),
        sg.In(size=(50, 1), enable_events=True, key="-VIDEOOUT-"),
        sg.FolderBrowse(),],
    [sg.Text("Nazwa wynikowego pliku wideo: ", size=(30,1)),
        sg.In(size=(50, 1), key="-NVIDEOOUT-"),],
    [sg.Text("Ścieżka do wynikowego pliku z logami:", size=(30,1)),
        sg.In(size=(50, 1), enable_events=True, key="-LOGOUT-"),
        sg.FolderBrowse(),],
    [sg.Text("Nazwa wynikowego pliku z logami: ", size=(30,1)),
        sg.In(size=(50, 1), key="-NLOGOUT-"),],
    [sg.Column([[sg.Button("Rozpocznij analizę")]], vertical_alignment='center', justification='center')],
    [sg.Column([[sg.Button("Otwórz plik z logami"), sg.Button("Otwórz plik wideo")]],
               vertical_alignment='center', justification='center')]
           ]

# Create the window
window = sg.Window("Analiza kolejki", layout, resizable=False)

videopath = ''
logpath = ''

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window
    if event == sg.WIN_CLOSED:
        break
    #Start anlysis
    if event == "Rozpocznij analizę":
        if os.path.join(values["-VIDEOIN-"]).lower().endswith((".mp4",".avi")):
            if values["-NLOGOUT-"]:
                logpath = os.path.join(values["-LOGOUT-"], values["-NLOGOUT-"] + '.txt')
            else:
                logpath = os.path.join(values['-LOGOUT-'], 'log' + str(random.randint(0, 100000)) + '.txt')
            errorWindow = sg.Window("Błąd", [
                [sg.Text(logpath), ]], resizable=False)
            errorWindow.read()
            if values["-NVIDEOOUT-"]:
                videopath = os.path.join(values["-VIDEOOUT-"][:-1], values["-NVIDEOOUT-"] + ".mp4")
            else:
                videopath = os.path.join(values["-VIDEOOUT-"][:-1], 'output' + str(random.randint(0, 100000)) + '.txt')
            errorWindow = sg.Window("Błąd", [
                [sg.Text(videopath), ]], resizable=False)
            errorWindow.read()
        else:
            errorWindow = sg.Window("Błąd", [
                [sg.Text("Błędna ścieżka do pliku!"), ]], resizable=False)
            errorWindow.read()
    #open log file
    if event == "Otwórz plik z logami":
        os.startfile(os.path.join(logpath))
    #open output video
    if event == "Otwórz plik wideo":
        os.startfile(os.path.join(values[videopath] ))

window.close()