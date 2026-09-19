# Oriđiđi

Procjena vjerojatnosti da je tekst na **hrvatskome jeziku** strojno napisan.

Sustav nikada ne daje presudu. Za svaki tekst vraća kalibriranu vjerojatnost, interval
nesigurnosti i jednu od tri oznake — *vjerojatno ljudski*, *neodlučno*, *vjerojatno strojno* —
a kada za procjenu nema osnove, suzdržava se umjesto da pogađa.

Nastalo kao prototip uz diplomski rad na Fakultetu organizacije i informatike u Varaždinu.

> ### Ovo nije alat za nadzor studenata
>
> Na akademskoj ljudskoj prozi sustav pogrešno označi **oko 3 % tekstova**, dakle otprilike
> svaki trideseti rad. Nije provjereno je li ta stopa jednaka za sve skupine autora, pa se mora
> pretpostaviti da nije. Alat smije biti pomoćno sredstvo u postupku koji ima ljudsku provjeru i
> pravo na odgovor. **Ne smije biti jedini dokaz protiv čovjeka.** Vidjeti
> [docs/ETHICS.md](docs/ETHICS.md).

---

## Brza instalacija

Potreban je Python 3.11 i oko 3 GB prostora (uz napredni način rada oko 7 GB).

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
powershell -ExecutionPolicy Bypass -File start.ps1
```

**Linux i macOS**

```bash
bash install.sh
./start.sh
```

`start` pokreće poslužitelj, čeka da se model učita i otvara sučelje na
`http://127.0.0.1:8000`.

## Dva načina rada

| | Osnovni | Napredni |
|---|---|---|
| Što radi | dorađeni enkoder za južnoslavenske jezike | enkoder uz dva statistička detektora bez nadzora |
| Trajanje analize | **0,41 s** | 28 s |
| Radna memorija | 4,9 GB | 8,5 GB |
| Veličina preuzimanja | 423 MB | uz to oko 3,4 GB |
| Stanje | **isporučeno** | istraživački |

Osnovni način je ono što se preporučuje i što je ocijenjeno kao proizvod. Napredni način bolje
rangira tekstove generatora koje sustav nije vidio, ali tu prednost ne pretvara u bolju radnu
točku, traje gotovo minutu po tekstu i nije prošao provjeru podudarnosti između različitoga
sklopovlja — zato je označen kao istraživački.

## Izmjerena izvedba

Sve vrijednosti potječu iz zapečaćenoga ocjenjivanja nad 2.328 zapisa koji tijekom razvoja
nisu korišteni.

| Mjera | Akademski tekst | Novinski tekst |
|---|---|---|
| Razlučivost (AUROC), agregat svih kohorti | 0,9267 | 0,9579 |
| Stopa lažno pozitivnih | 3,00 % | 0,20 % |
| Kalibracijska pogreška (ECE) | 0,0025 | |

Razlučivost na tekstovima iste vrste kakvi su korišteni u učenju iznosi preko 0,99, ali ta
vrijednost ne opisuje ponašanje na novome tekstu i ovdje se namjerno ne navodi kao izvedba
sustava.

**Gdje se lomi:** na generatoru koji nije vidio, na akademskome tekstu, na tekstu kraćem od
donje granice i pri promjeni sklopovlja.

## Kako radi

1. Normalizacija znakova, uključujući preslikavanje homoglifa. Mjereno je da upravo taj
   jeftini korak zatvara napad zamjenom vizualno istovjetnih znakova.
2. Vrata jezika i vrata dužine. Tekst kraći od **100 riječi** ili koji nije prepoznat kao
   hrvatski ne dobiva procjenu, nego obrazloženje.
3. Ocjenjivanje: enkoder, a u naprednom načinu uz njega i Binoculars i Fast-DetectGPT nad
   zajedničkim parom jezičnih modela.
4. Kalibracija u vjerojatnost i konformna predikcija, koja daje zonu suzdržavanja.
5. Prikaz: vjerojatnost, interval, oznaka i tragovi po riječima.

Podrobnije u [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Model i podatci

Enkoder je dorađen iz [BERTića](https://huggingface.co/classla/bcms-bertic) na skupu od
11.226 zapisa hrvatskoga teksta iz javno dostupnih izvora, uz strojnu stranu koju je
generiralo sedam modela. Težine se ne nalaze u ovome repozitoriju — preuzima ih instalacijska
skripta iz izdanja (*release*) i provjerava im sažetke.

Za napredni način instalacijska skripta traži vaš **Hugging Face token**, preuzima
`google/gemma-2-2b` iz Googleova repozitorija i kvantizira ga na vašem računalu. Time uvjete
uporabe modela Gemma prihvaćate izravno s Googleom; ovaj projekt te težine ne dijeli.

Vidjeti [docs/MODEL_CARD.md](docs/MODEL_CARD.md) i [docs/LICENSES.md](docs/LICENSES.md).

## Što sustav ne radi

* Ne otkriva strojno **preveden** tekst i na tome nije mjeren.
* Ne mjeri mješovito autorstvo, dakle tekst koji je čovjek napisao a model uredio.
* Ne pokazuje koji je dio dokumenta strojni; ta je funkcionalnost uklonjena jer mjerenje nije
  pokazalo dovoljnu pouzdanost.
* Nije uspoređen s objavljenim detektorima na istim podatcima, pa se ne tvrdi da je bolji ni
  lošiji od njih.

## Razvoj

```bash
uv sync --extra serve --extra encoder    # okruženje
uv run pytest -q tests/browser           # sučelje, radi i bez težina
uv run ruff check . && uv run mypy src   # provjere
```

Provjera u pregledniku pokreće pravu aplikaciju s lažnim modelima, pa je korisna i prije nego
što su težine preuzete.

## Licencija

Izvorni kod: [MIT](LICENSE). Težine modela i komponente drugih autora imaju vlastite uvjete,
popisane u [docs/LICENSES.md](docs/LICENSES.md).
