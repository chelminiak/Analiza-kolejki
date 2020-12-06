import os
import PySimpleGUI as sg
from object_tracker import find
import random
import time

disable = True
layout = [
    [sg.Text("Ścieżka do pliku wejściowego:", size=(30, 1)),
     sg.In(size=(50, 1), disabled=True, enable_events=False, key="-VIDEOIN-"),
     sg.FileBrowse()],
    [sg.Text("Ścieżka do wynikowego pliku video: ", size=(30, 1)),
     sg.In(size=(50, 1), disabled=True, enable_events=False, key="-VIDEOOUT-"),
     sg.FolderBrowse(), ],
    [sg.Text("Nazwa wynikowego pliku wideo: ", size=(30, 1)),
     sg.In(size=(50, 1), enable_events=False, key="-NVIDEOOUT-"), ],
    [sg.Text("Ścieżka do wynikowego pliku z logami:", size=(30, 1)),
     sg.In(size=(50, 1), disabled=True, enable_events=True, key="-LOGOUT-"),
     sg.FolderBrowse(), ],
    [sg.Text("Nazwa wynikowego pliku z logami: ", size=(30, 1)),
     sg.In(size=(50, 1), key="-NLOGOUT-"), ],
    [sg.Column([[sg.Button("Rozpocznij analizę")]], vertical_alignment='center', justification='center')],
    [sg.Column(
        [[sg.Button("Otwórz plik z logami", disabled=disable), sg.Button("Otwórz plik wideo", disabled=disable)]],
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
    # Start anlysis
    if event == "Rozpocznij analizę":
        if os.path.join(values["-VIDEOIN-"]):
            if values["-NLOGOUT-"]:
                logpath = os.path.join(values["-LOGOUT-"], values["-NLOGOUT-"] + '.txt')
            else:
                logpath = os.path.join(values['-LOGOUT-'], 'log' + str(random.randint(0, 100000)) + '.txt')
            if values["-NVIDEOOUT-"]:
                videopath = os.path.join(values["-VIDEOOUT-"], values["-NVIDEOOUT-"] + '.avi')
            else:
                videopath = os.path.join(values["-VIDEOOUT-"], 'output' + str(random.randint(0, 100000)) + '.avi')
            infoWindow = sg.Window("Analiza w toku...", [
                [sg.Text("Po zamknięciu tego okna program stanie się nieresponsywny. Po zakończeniu przetwarzania wyświetlony zostanie odpowiedni komunikat"), ]], resizable=False)
            infoWindow.read()
            analyze = find.findPerson(values["-VIDEOIN-"], videopath, logpath)
            while analyze != 0 and analyze !=1:
                infoWindow.refresh()
            if analyze == 0:
                infoWindow = sg.Window("Analiza zakończona!", [
                    [sg.Text("Przetwarzanie zakończone sukcesem!"), ]], resizable=False)
                infoWindow.read()
                disable = False
                window["Otwórz plik z logami"].update(disabled=disable)
                window["Otwórz plik wideo"].update(disabled=disable)
            if analyze == 1:
                errorWindow = sg.Window("Błąd", [
                    [sg.Text("Niepowodzenie analizy"), ]], resizable=False)
                errorWindow.read()
        else:
            errorWindow = sg.Window("Błąd", [
                [sg.Text("Błędna ścieżka do pliku!"), ]], resizable=False)
            errorWindow.read()
    # open log file
    if event == "Otwórz plik z logami":
        os.startfile(os.path.join(logpath))
    # open output video
    if event == "Otwórz plik wideo":
        os.startfile(os.path.join(videopath))

window.close()
