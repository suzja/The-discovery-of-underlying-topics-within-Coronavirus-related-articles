#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-,

import string
import re
def convert(lst):
    return (lst[0].split())


if __name__ == '__main__':

    lijst = [" Afgelopen; 2. Week; 3. Ziekenhuizen; 4. Coronavirus; 5. IC; 6. Etmaal;\\ 7. Zondag; 8. Ziekenhuis; 9. Europese; 10. Meldingen; 11. Doses; 12. Tests;\\ 13. Meldt; 14. Gemeld; 15. Opgenomen; 16. Ontbreken; 17. Pfizer; 18. Mogelijk;\\ 19. Premier; 20. Daling; 21. Reguliere; 22. Zorg; 23. Zuid; 24. Februari; \\ 25. Getest; 26. Lockdown; 27. Versoepelingen; 28. Hoogste; 29. Bekend; 30. Gaan.\end{tabular} \\ \hline 1. Aantal; 2. Jaar; 3. Procent; 4. Patienten; 5. Kabinet; 6. Dag; 7. LCPS; 8. Cijfers;\\ 9. Politie; 10. Intensive; 11. Euro; 12. Zegt; 13. EU; 14. Sterfgevallen; 15. Sinds;\\ 16. Wel; 17. Elementen; 18. Onderzoek; 19. Overleden; 20. President; 21. Janssen;\\ 22. Tweede; 23. Maart; 24. Vertraging; 25. Blijkt; 26. Toegenomen; 27. Jongeren;\\ 28. Zei; 29. Middel; 30. Curacao.\end{tabular} \\ \hline 1. Vaccin; 2. Rivm; 3. Positieve; 4. Covid; 5. Vaccins; 6. Nederland; 7. Minder; \\ 8. Vrijdag; 9. Liggen; 10. Weer; 11. Astrazeneca; 12. Maandag; 13. Jonge; \\ 14. Besmettingen; 15. Britse; 16. Care; 17. April; 18. Ema; 19. Printversie; \\ 20. Afdelingen; 21. Corona; 22. Coronacrisis; 23. Mogen; 24. Gemeente; 25. Positief;\\ 26. Amsterdam; 27. Duitse; 28. Bezoekers; 29. Geregistreerd; 30. Kamer.\end{tabular} \\ \hline 1. Nieuwe; 2. Miljoen; 3. Eerste; 4. Eerder; 5. Variant; 6. Uur; 7. Woensdag;\\ 8. Coronatests; 9. Virus; 10. Land; 11. Per; 12. Twee; 13. Vanaf; 14. Werden; \\ 15. Minister; 16. Artikel; 17. Miljard; 18. Duitsland; 19. Ruim; 20. Tussen; 21. Testen;\\ 22. Totaal; 23. Momenteel; 24. Prik; 25. Liveblog; 26. Dagen; 27. Open; 28. Besmet;\\ 29. Ministerie; 30. Vaak.\end{tabular} \\ \hline 1. Mensen; 2. Volgens; 3. Coronapatienten; 4. Dinsdag; 5. Zaterdag; 6. Donderdag;\\ 7. Landelijk; 8. Spreiding; 9. GGD; 10. Rutte; 11. Landen; 12. Moeten; \\ 13. Nederlandse; 14. Avondklok; 15. NB; 16. Januari; 17. Demissionair; 18. Vorig; \\ 19. Doorgegeven; 20. WHO; 21. OMT; 22. Opzichte; 23. Tijdens; 24. Scholen; 25. Alle; \\ 26. Bedrijven; 27. Weken; 28. Versie; 29. Eiland; 30 Rijksinstituut."]
    newlist = []

    with open('negalexicon.txt', "r+") as file1:
        fileline1= file1.readlines()
        lijst = convert(lijst)
        for x in lijst: # <--- Loop through the list to check
            x = x.lower()
            # x = convert(x)
            # re.sub(r'[^\w\s]','',x)
            # re.sub("^[0-9,;]+$", '', x)
            x = x.replace(';', '')
            x = x.replace('.', '')
            # print(x)
            #exit(1)
            # print(x)
            # print("klaarz")
            for line in fileline1: # <--- Loop through each line
                line = line.replace('\n', '')
                # print(x)
                # print(line)
                if line == x:
                    # print(x)
                    newlist.append(x)


newlist = list(set(newlist))
print(newlist)
print(len(newlist))
