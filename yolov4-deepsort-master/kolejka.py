class Kolejka:
    def __init__(self, liczbaOsob):
        self.liczbaOsob = liczbaOsob
        self.listaOsob = []

    def dodajOsobe(self, osoba):
        self.listaOsob.append(osoba)

    def usunOsobe(self, osoba):
        self.listaOsob.remove(osoba)

    def getLiczbaOsob(self):
        return self.liczbaOsob