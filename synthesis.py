import pandas as pd
import torch

class Feature_Vector:
    def __init__(self):
        self.conductivity = {
            'Konduktivitas<=10': None,
            '10<Konduktivitas<=30': None,
            '30<Konduktivitas<=50': None,
            '50<Konduktivitas': None
        }

        self.magnetic = {
            'Ferromagnetik': None,
            'Diamagnetik': None,
            'Paramagnetik': None
        }

        self.mass_and_melting_point = {
            'Massa>10': None,
            '10<Massa<20': None,
            '20<Massa': None,
            'TitikLebur<0': None,
            'TitikLebur>500': None,
            '500<TitikLebur<1000': None,
            '1000<TitikLebur<1500': None,
            '1500<TitikLebur': None
        }

        self.mechanical_properties = {
            'Kekekerasan<=2': None,
            '2<Kekerasan<=4': None,
            '4<Kekerasan<=6': None,
            '6<Kekerasan<=8': None,
            '8<Kekerasan': None,
            'Elastisitas<=1': None,
            '1<Elastisitas<=100': None,
            '100<Elastisitas<=200': None,
            '200<Elastisitas<=300': None,
            '300<Elastisitas': None
        }

        self.corrosion_and_forging = {
            'STinggiTempa': None,
            'TinggiTempa': None,
            'SedangTempa': None,
            'RendahTempa': None,
            'TidakTempa': None,
            'STinggiKarat': None,
            'TinggiKarat': None,
            'SedangKarat': None,
            'RendahKarat': None,
            'SRendahKarat': None
        }
        self.data = pd.read_csv('hot_encoded_resource.csv').drop(columns='#')
        

    def deduce_conductivity(self):
        # Fungsi deduksi untuk konduktivitas
        for key in self.conductivity:
            choice = input(f'Apakah logam memiliki {key}? (yes/no): ').strip().lower()
            if choice == 'yes':
                self.conductivity[key] = 1
                self.data = self.data[self.data[key] == 1]
            elif choice == 'no':
                self.conductivity[key] = 0
                self.data = self.data[self.data[key] == 0]

    def deduce_magnetic(self):
        # Fungsi deduksi untuk sifat magnetik
        for key in self.magnetic:
            choice = input(f'Apakah logam bersifat {key}? (yes/no): ').strip().lower()
            if choice == 'yes':
                self.magnetic[key] = 1
            elif choice == 'no':
                self.magnetic[key] = 0
            else:
                print('Harap jawab dengan "yes" atau "no".')

    def deduce_mass_and_melting(self):
        # Fungsi deduksi untuk massa dan titik lebur
        for key in self.mass_and_melting_point:
            choice = input(f'Apakah logam sesuai dengan kategori {key}? (yes/no): ').strip().lower()
            if choice == 'yes':
                self.mass_and_melting_point[key] = 1
            elif choice == 'no':
                self.mass_and_melting_point[key] = 0
            else:
                print('Harap jawab dengan "yes" atau "no".')

    def predict_vector(self):
        self.deduce_general()
        self.deduce_magnetic()
        self.deduce_mass_and_melting()
        
        vector = []
        vector.extend(list(self.conductivity.values()))
        vector.extend(list(self.magnetic.values()))
        vector.extend(list(self.mass_and_melting_point.values()))
        vector.extend(list(self.mechanical_properties.values()))
        vector.extend(list(self.corrosion_and_forging.values()))

        return torch.tensor(vector)
