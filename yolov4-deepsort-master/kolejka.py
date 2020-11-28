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
        liczbaOsob=0
        for osoba in self._listaOsob:   
            liczbaOsob+=1
        return liczbaOsob

    def getOsoba(self, trackId):
        for osoba in self._listaOsob:
            if osoba.getTrackId() == trackId:
                return osoba
        return None

    def getListaOsob(self):
        return self._listaOsob

    def getLiczbaOsobKategorie(self):
        liczbaOsobKategorie = [0, 0, 0]
        for osoba in self._listaOsob:
            if osoba.getKategoria() == 0:
                liczbaOsobKategorie[0] += 1
            if osoba.getKategoria() == 1:
                liczbaOsobKategorie[1] += 1
            if osoba.getKategoria() == 2:
                liczbaOsobKategorie[2] += 1
        return liczbaOsobKategorie