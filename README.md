# Oriđiđi

Procjena vjerojatnosti da je tekst na **hrvatskome jeziku** strojno napisan.

Sustav ne utvrđuje autorstvo. Za prihvaćen tekst odgovor sadrži kalibriranu procjenu,
konformni predikcijski skup i iz njega izvedenu oznaku: ljudski, strojni ili suzdržano.
Suzdržavanje uvijek proizlazi iz praznoga skupa (dvočlanih skupova nema); tada sučelje
kaže da procjena ne podupire nijednu oznaku. Kada tekst ne zadovolji uvjete za analizu,
sustav vraća obrazloženje bez procjene.

Nastalo kao prototip uz diplomski rad na Fakultetu organizacije i informatike u Varaždinu.

> ### Ovo nije alat za nadzor studenata
>
> Na akademskoj ljudskoj prozi sustav pogrešno označi **oko 3 % tekstova**, dakle otprilike
> svaki trideseti rad. Nije provjereno je li ta stopa jednaka za sve skupine autora, pa se mora
> pretpostaviti da nije. Rezultat nije dokaz i **ne smije biti osnova za sankciju ni za odluku o
> osobi.** Vidjeti
> [docs/ETHICS.md](docs/ETHICS.md).

---

## Instalacija i pokretanje

Arhiva sadrži aplikacijski izvorni kod i testove. Ne sadrži korpuse ni istraživačke
skripte. Težine modela nisu u arhivi: dorađeni BERTić i kalibracije (25 datoteka, oko
423 MiB) preuzimaju se pri instalaciji s adrese `base_url` iz `release_manifest.json`, a
njihovi SHA-256 sažeci provjeravaju se prema istoj datoteci. Težine Gemma se ne
redistribuiraju (Gemma Terms of Use). Adresa se može zadati i parametrom `-BaseUrl`.

Potrebni su Python 3.11 i alat `uv` (instalacijska skripta za Windows po potrebi ga
instalira preko winget-a).

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
powershell -ExecutionPolicy Bypass -File start.ps1
```

`install.ps1` ima šest koraka: provjera alata `uv`, Python okruženje i ovisnosti,
mape, hrvatski rječnik, provjera SHA-256 sažetaka artefakata u `models/` i
**neobavezni korak 6 (napredni način)**. Korak 6 traži Hugging Face token, preuzima
`google/gemma-2-2b` i `google/gemma-2-2b-it` iz Googleova repozitorija, prevodi
llama.cpp (oznaka b10516) i kvantizira modele u Q4_K_M. Prije preuzimanja provjerava
preduvjete: git, CMake, Visual Studio Build Tools s C++ alatima i barem 14 GB
slobodnoga prostora. Parametri skripte:

* `-PreskociNapredni` preskače korak 6 bez pitanja (samo osnovni način);
* `-HfToken <token>` zadaje token (ili varijabla okoline `HF_TOKEN`); s tokenom se
  korak 6 izvodi bez pitanja.

Ako token nije zadan i parametar `-PreskociNapredni` nije naveden, skripta ga traži;
prazan unos preskače napredni način. Skripta se smije pokrenuti više puta.

`start.ps1` pokreće lokalni poslužitelj, čeka da se modeli učitaju i otvara
`http://127.0.0.1:8000`. Napredni način dostupan je samo ako je korak 6 dovršen.

**Linux i macOS**

```bash
bash install.sh
./start.sh
```

Te skripte pripremaju samo osnovni način.

**Testovi**

```bash
uv sync --locked
uv run pytest -q -m "not slow"
```

## Dva načina rada

| | Osnovni | Napredni |
|---|---|---|
| Što radi | BERTić (ONNX FP32) čita prvih 160 tokena | Gemma-2-2B Q4_K_M (Binoculars i Fast-DetectGPT) uz enkoder, logistička fuzija |
| Medijan trajanja analize (CPU prijenosnika) | **0,12 s** (P90 0,129 s) | 27,8 s (P90 oko 30 s) |
| Radna memorija | oko 5,0 GB (proces s učitana oba načina) | do 8,1 GB nakon prvoga naprednog zahtjeva |
| Isporuka | preuzima se korakom 5 skripte `install.ps1` | instalira se korakom 6 skripte `install.ps1` |
| Evaluacija | GPU i CPU | **samo GPU** |
| Stanje | implementiran | istraživački |

Napredni način kalibriran je i evaluiran isključivo na izvedbi na grafičkom procesoru.
Prijenos njegove kalibracije na procesor nije uspio (najveća razlika procjene 0,3421), pa se
naprednim rezultatima na procesoru ne smije pripisivati ista pouzdanost. Osnovni je način
potvrđen i na procesoru. Napredni način poboljšava ukupno rangiranje na suvremenim
akademskim tekstovima, ali prednost pri strogoj radnoj točki nije potvrđena.

Parametri po načinu rada (konformni skup Mondrian, α = 0,10):

| | Osnovni | Napredni |
|---|---|---|
| Ulaz enkodera | prvih 160 tokena | prvih 160 tokena |
| Kalibracija | izotonična | logistička fuzija, zasebna kalibracija |
| Kvantil q0 (ljudski razred) | 0,0800 | 0,0411 |
| Kvantil q1 (strojni razred) | 0,0833 | 0,0536 |
| Radni prag | 0,889 | 0,932 |

Najmanja duljina teksta u zadanoj konfiguraciji je 100 riječi.

## Izmjerena izvedba

Sve vrijednosti odnose se na osnovni način i potječu iz ocjenjivanja na završnom skupu od
2.328 zapisa koji tijekom razvoja nisu korišteni. Agregat po domeni čini 836 ljudskih i
164 strojna zapisa.

| Mjera | Akademski tekst | Novinski tekst |
|---|---|---|
| Razlučivost (AUROC), agregat svih kohorti | 0,9267 | 0,9579 |
| Stopa lažno pozitivnih (na svih 1.000 ljudskih zapisa po domeni) | 3,00 % | 0,20 % |
| Kalibracijska pogreška (ECE), agregat | 0,0575 | 0,0213 |

Na zasebnoj lokalnoj kohorti starijih generatora AUROC je iznad 0,99, ali rezultat se
ne prenosi izravno na suvremene generatore. Primjerice, osnovni način na suvremenim
akademskim tekstovima ima AUROC 0,8759 i TPR 0,19 pri 1 % lažno pozitivnih.

Rezultat ovisi o generatoru, domeni, duljini i sklopovlju. Napredna kalibracija na
procesoru nije potvrđena, zbog čega taj način ostaje istraživački.

## Kako radi

1. Normalizacija znakova, uključujući preslikavanje homoglifa. Mjereno je da upravo taj
   jeftini korak ublažava ispitani napad zamjenom vizualno istovjetnih znakova.
2. Vrata jezika i vrata dužine. Tekst kraći od **100 riječi** ili koji nije prepoznat kao
   hrvatski ne dobiva procjenu, nego obrazloženje.
3. Ocjenjivanje: enkoder čita prvih 160 tokena, a u naprednom načinu uz njega i
   Binoculars i Fast-DetectGPT nad zajedničkim parom jezičnih modela.
4. Kalibracija u vjerojatnost i konformna predikcija; prazan skup znači suzdržavanje.
5. Prikaz: kalibrirana procjena, konformni skup, oznaka i relativni tragovi tokena u
   naprednom načinu.

Podrobnije u [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Model i podatci

Razvojni repozitorij imao je 11.226 zapisa različitih uloga. Završni enkoder
[BERTić](https://huggingface.co/classla/bcms-bertic) ponovno je učen na **896 parova,
odnosno 1.792 zapisa**; kalibracijski i završni evaluacijski zapisi nisu bili dio
tog učenja. Dorađeni enkoder preuzima se u mapu `models/`.

Za napredni način instalacijska skripta (korak 6) traži vaš **Hugging Face token**,
preuzima Gemma-2-2B iz Googleova repozitorija i kvantizira ga na vašem računalu. Time uvjete
uporabe modela Gemma prihvaćate izravno s Googleom; ovaj projekt te težine ne dijeli.

Vidjeti [docs/MODEL_CARD.md](docs/MODEL_CARD.md) i [docs/LICENSES.md](docs/LICENSES.md).

## Što sustav ne radi

* Nije vrednovan na strojno prevedenom tekstu.
* Nije završno vrednovan na mješovitom autorstvu, primjerice tekstu koji je čovjek
  napisao, a model uredio.
* Ne pokazuje koji je dio dokumenta strojni; mjerenje nije pokazalo dovoljnu pouzdanost za
  takav prikaz.
* Nije uspoređen s objavljenim detektorima na istim podatcima, pa se ne tvrdi da je bolji ni
  lošiji od njih.

## Razvoj

```bash
uv sync --locked                          # okruženje za pregled i testove
uv run pytest -q -m "not slow"           # brzi testovi
uv run pytest -q tests/browser           # sučelje, radi i bez težina
uv run ruff check . && uv run mypy src   # provjere
```

Provjera u pregledniku pokreće pravu aplikaciju s lažnim modelima.

Podrijetlo podataka, modela i koda: [PROVENANCE.md](PROVENANCE.md).

## Licencija

Izvorni kod: [MIT](LICENSE). Težine modela i komponente drugih autora imaju vlastite uvjete,
popisane u [docs/LICENSES.md](docs/LICENSES.md).
