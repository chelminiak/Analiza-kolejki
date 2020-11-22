class Kolejka:
    def __init__(self, liczbaOsob):
        self._liczbaOsob = liczbaOsob
        self._listaOsob = []

    def dodajOsobe(self, osoba):
        self._listaOsob.append(osoba)
        self._liczbaOsob += 1

    def usunOsobe(self, osoba):
        self._listaOsob.remove(osoba)
        self._liczbaOsob -= 1

    def getLiczbaOsob(self):
        return self._liczbaOsob

    def getOsoba(self, trackId):
        for osoba in self._listaOsob:
            if osoba.getTrackId() == trackId:
                return osoba
        return None

    def getListaOsob(self):
        return self._listaOsob
