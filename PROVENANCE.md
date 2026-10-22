# Podrijetlo podataka, modela i koda

Ovaj dokument navodi odakle potječu podatci, modeli i kod koje Oriđiđi koristi i pod
kojim licencijama. Podrobniji popis licencija je u [docs/LICENSES.md](docs/LICENSES.md).

## Što arhiva sadrži

Aplikacijski izvorni kod, konfiguracije, web-sučelje i testove. Model i kalibracije
preuzimaju se pri instalaciji (`release_manifest.json`). Arhiva ne sadrži korpuse, generirane tekstove, skripte
učenja i evaluacije ni težine modela Gemma. Znanstveni rezultati opisani su u završnom
radu; testovi u ovoj arhivi ne ponavljaju završne eksperimente.

## Podatci

Podatci korišteni za učenje, kalibraciju i evaluaciju ne distribuiraju se s ovim kodom.

| Izvor | Uloga | Licencija |
|---|---|---|
| MaCoCu-Genre.hr (11356/1969) | ljudski tekstovi, novinska domena | CC0 |
| ZNANJE / dabar_hr (`procesaur/znanje`) | ljudski tekstovi, akademska domena | CC-BY-SA-4.0 |
| Metapodatci repozitorija Dabar (OAI-PMH) | godine objave akademskih radova | javni metapodatci |

Strojni tekstovi generirani su otvorenim modelima i modelima dostupnima preko API-ja;
uvjeti uporabe pojedinoga generatora navedeni su u [docs/LICENSES.md](docs/LICENSES.md).

## Modeli

* **Enkoder.** Osnova je [BERTić](https://huggingface.co/classla/bcms-bertic)
  (`classla/bcms-bertic`, Apache-2.0), dorađen kao binarni klasifikator na podatcima
  navedenima gore i izvezen u ONNX (FP32). Dorađeni model i kalibracije preuzimaju se u `models/`.
* **Gemma-2-2B** (`google/gemma-2-2b` i `google/gemma-2-2b-it`, Gemma Terms of Use) koristi
  se u naprednom načinu. Težine se ne distribuiraju: `install.ps1` ih preuzima s
  Hugging Facea s korisnikovim tokenom, na točno određenim revizijama, i kvantizira ih
  lokalno u Q4_K_M alatom llama.cpp (oznaka b10516, MIT). Time korisnik uvjete uporabe
  prihvaća izravno s Googleom.
* Binoculars (BSD-3-Clause) i Fast-DetectGPT (MIT) ponovno su implementirani prema
  objavljenim opisima; izvorni kod nije kopiran. Projekt `mgt-detection-benchmark`
  (GPL-3.0) poslužio je samo kao metodološka referenca, bez preuzetoga koda.

## Kod i ostale komponente

* Izvorni kod aplikacije napisao je autor i objavljen je pod licencijom [MIT](LICENSE).
* Hrvatski rječnik (hunspell hr_HR) preuzima `tools/fetch_lexicons.py` pri instalaciji;
  nije dio arhive. Licencija: GPL-2.0 / LGPL-2.1 / MPL-1.1.
* Karta zamjene vizualno istovjetnih znakova izvedena je iz Unicode UTS #39
  (`confusables.txt`, Unicode License).
* Pomoćne biblioteke: spylls (MIT), lingua-language-detector (Apache-2.0) i neobavezno
  CLASSLA-Stanza (Apache-2.0); popis ovisnosti zaključan je u `uv.lock`.
* Fontovi Archivo, IBM Plex Sans i IBM Plex Mono (SIL Open Font License 1.1) poslužuju se
  lokalno. Logotip je autorsko djelo autora rada.
