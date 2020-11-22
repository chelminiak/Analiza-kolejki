class Osoba:

    # kategorie
    # 0 - red
    # 1 - green
    # 2 - blue

    def __init__(self, kategoria, start, koniec, pozycjaX, pozycjaY):
        self.kategoria = kategoria
        self.start = start
        self.koniec = koniec
        self.pozycjaX = pozycjaX
        self.pozycjaY = pozycjaY

    def getKategoria(self):
        return self.kategoria

    def getStart(self):
        return self.start

    def getKoniec(self):
        return self.koniec

    def getPozycjaX(self):
        return self.pozycjaX

    def getPozycjaY(self):
        return self.pozycjaY

