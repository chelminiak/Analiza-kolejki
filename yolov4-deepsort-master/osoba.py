class Osoba:

    # kategorie
    # 0 - red
    # 1 - green
    # 2 - blue

    def __init__(self, trackId, kategoria, start, koniec, pozycjaX, pozycjaY):
        self._trackId = trackId
        self._kategoria = kategoria
        self._start = start
        self._koniec = koniec
        self._pozycjaX = pozycjaX
        self._pozycjaY = pozycjaY

    def getKategoria(self):
        return self._kategoria

    def getStart(self):
        return self._start

    def getKoniec(self):
        return self._koniec

    def getPozycjaX(self):
        return self._pozycjaX

    def getPozycjaY(self):
        return self._pozycjaY

    def setKoniec(self, koniec):
        self._koniec = koniec

    def getTrackId(self):
        return self._trackId
