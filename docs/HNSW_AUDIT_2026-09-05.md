# HNSW-CLJ: életciklus, hiányosságok és indexépítési kutatás

Dátum: 2026-09-05. Kiinduló commit: `b7ee89f`.

> Történeti audit, az alábbi hiánylista a library-refaktor **előtti** állapotot írja le.
> Azóta a `hnsw.api` kezelt indexet ad: bemenetellenőrzés, olvasó/író zárolás,
> újraépítéses frissítés és törlés, `clear!`/`close!`, verziózott atomi mentés került bele.
> A frissítés és törlés azóta lokális gráfjavítást használ; a kompakt keresési
> snapshot továbbra is teljesen frissül. Részletek: `LOCAL_MAINTENANCE.md`.
> A régi API-k és alternatív algoritmusok eltávolításra kerültek.
> Az aktuális felület és korlátok a gyökér README-ben találhatók.
> Az itt szereplő teljesítményadatok a belső HNSW-kernelre vonatkoznak,
> nem tartalmazzák az új kezelt API bemenetellenőrzési költségét.

A vizsgálat a `hnsw.ultra-fast` építőre, a Java keresési kernelekre, a
`hnsw.api` és `hnsw.api.simple` felületekre, valamint a perzisztenciára és
az ezekhez tartozó tesztekre összpontosít. A régi alternatív ANN-megvalósításokat
és a `wip` könyvtárat ez nem minősíti teljes körűen.

## Mi működik, és mi hiányzik?

| Művelet | Jelenlegi állapot | Következmény / következő lépés |
| --- | --- | --- |
| Teljes index építése | Működik, egy közös gráfon, több szállal | Új, választható float32 építési gyorsítás készült |
| Új ID beszúrása | `api/add-vector!`, `ultra/insert-single`, `insert-batch` | A következő keresés újraépítheti a teljes kompakt keresési snapshotot |
| Létező ID frissítése | Nincs; az ismételt ID kivételt okoz | Vektort és a hozzá tartozó éleket együtt kell frissíteni |
| Egy vektor logikai törlése | Nincs `mark-deleted!` / `unmark-deleted!` | A keresési algoritmusnak továbbra is át kell tudnia haladni a törölt csomóponton |
| Fizikai törlés / tárhely-visszanyerés | Nincs | Első lépésként az aktív vektorokból épített új gráf javasolt |
| Teljes memóriabeli index eldobása | A hivatkozások elengedése után JVM GC | A JVM később szabadítja fel; nincs explicit `close!` / `clear!` API |
| Mentett indexfájl törlése | Normál fájlrendszeri művelet | Nem módosítja a már betöltött indexet; nem HNSW-algoritmikus művelet |
| Mentés/betöltés a fő API-n | Korábban kivételt dobó helykitöltő; most bekötve | Az új fájlok megőrzik a beépített metrikát és mindkét precíziót |
| Egyidejű olvasás | Kész kompakt snapshoton támogatott | A Java munkaterület szálanként külön áll rendelkezésre |
| Egyidejű írás és keresés ugyanazon a gráfon | Nincs biztonságos konzisztenciaszerződés | Külön felépített gráf atomikus cseréje javasolt |
| Szűrés / törölt ID-k kihagyása a keresés során | A fő motorban nincs | A protokoll definíciója önmagában nem jelent implementációt |
| Metaadatok tartós tárolása a simple API-n | Nincs; a mentés csak a gráfot írja | Betöltéskor a metadata üres, a size 0, a dimensions nil lesz |
| Bináris, verziózott, atomikus mentés | Nincs; EDN `spit`/`slurp` | Nagy indexnél memória- és I/O-költség, felülíráskor nincs crash-atomic csere |
| Dimenzió és numerikus input ellenőrzése | Részleges | A hibás beszúrást még a gráf módosítása előtt kell elutasítani |
| Automatikus `efSearch` | Nincs adaptív vezérlő a fő motorban | Fix alapérték és lekérdezésenkénti felülírás van |

Az eredeti HNSW leírás a réteges gráfot, a beszúrást, a keresést és a
szomszédválasztást adja. A teljes CRUD, a fájlkezelés és a konkurens
tranzakciós működés külön implementációs feladat. A
[hnswlib dokumentációja](https://github.com/nmslib/hnswlib#api-description)
ehhez már kínál azonos ID-n frissítést, törlési jelölést, visszaállítást és
törölt helyek újrafelhasználását, de az `add_items` és `knn_query` együttes
futtatását ott sem dokumentálják szálbiztosnak.

## Fontos kódhibák és hiányok, prioritási sorrendben

1. **P1: snapshot építése módosuló gráfból.** Az `insert-single-at-level`
   a beszúrás elején érvényteleníti a snapshotot. Az `optimize-search!` eközben
   rögzítheti a még változó gráfot; az offsetek és a szomszédlisták külön
   bejárásból készülnek. Nincs olyan verzióellenőrzés vagy közös zárolás,
   amely garantálná a kész és konzisztens snapshot publikálását.
   A meglévő mixed-operations teszt másik gráfot épít és atomot cserél,
   ezért ezt az ugyanazon gráfon fellépő versenyhelyzetet nem teszteli.
   Forrás: `src/hnsw/ultra_fast.clj`, `src/hnsw/search/compact.clj`,
   `test/hnsw/integration_test.clj`.

2. **P1: mellékhatás a simple API `swap!` függvényében.** Az `add!` a gráf
   beszúrását a CAS miatt újrafuttatható függvényen belül végzi. Konkurens
   metadata/size módosítás után ugyanaz a beszúrás megismétlődhet és hibát
   dobhat, miközben a gráf már megváltozott. A gráfmódosításnak egyszer kell
   lefutnia, jól meghatározott írási zárolás alatt.
   Forrás: `src/hnsw/api/simple.clj`.

3. **P1: input és részleges batch-hibák.** Nincs egységes, beszúrás előtti
   dimenzió-, NaN/Infinity- és batch-ID ellenőrzés. Egy rossz vektor vagy
   duplikált ID miatt a batch hibázhat, amikor a korábbi elemek már bekerültek.
   Egy euclidean gráf a beadott `double[]` hivatkozását is megtartja, ezért
   a hívó későbbi tömbmódosítása elronthatja a gráf és a távolságok összhangját.
   Ezt validálási és tulajdonlási szerződéssel kell rendezni.

4. **P1: törlés és upsert hiánya.** Egy csomópont egyszerű eltávolítása a
   `nodes` mapből lógó éleket és elérhetetlen részgráfokat okozhat. Ugyanazon
   ID vektorának egyszerű felülírása sem helyes: az élek a régi helyéhez
   tartoznak. Ezért egyik művelet sem kerülhet be puszta map-módosításként.

5. **P2: perzisztencia és API-k közötti eltérés.** A fő API mentés/betöltés
   helykitöltőit és a beépített metrika elvesztését ez a változtatás javítja.
   A simple API wrapperállapotának megőrzése, a bináris formátum, az atomikus
   fájlcsere és a véletlenszám-generátor folytatható állapota még hiányzik.
   A `hnsw.api.unified` jelenleg az `UltraGraph` típust sem illeszti az
   `ANNIndex` protokollhoz.

6. **P2: módosítás utáni teljes snapshot-másolás.** Egyetlen új vektor utáni
   első keresés is újra másolhatja a teljes vektor- és szomszédtárat.
   Ez keresési késleltetési tüskét okozhat gyakori írás mellett.
   Batch írás, explicit snapshot publikálás, később base+delta index javasolt.

7. **P2: benchmarkok értelmezése.** A régi Biblia `exact_verse_recall`
   azt méri, előkerül-e az azonos vers a másik nyelven. Ez hasznos alkalmazási
   metrika, de nem a teljes végigkeresés top-10 halmazához mért ANN Recall@10.
   Az új építési benchmark már az utóbbit számolja. A README régi
   „no third-party dependencies” állítása sem írja le a teljes `deps.edn`
   állományt; a mostani távolságkernel valóban JVM-es, miközben a projekt
   más megvalósításokhoz további függőségeket is felsorol.

## Hogyan történjen most egy törlés vagy frissítés?

Az alkalmazás tartsa meg a forrásvektorokat és metaadatokat az indexen kívül.
Módosítsa a forrásadatot, építsen belőle új indexet, majd az index és a
metaadatok közös állapotát egyetlen `reset!` hívással cserélje ki.
Az újraépítés alatt az olvasók használhatják a régi, változatlan példányt.
Az írókat sorba kell rendezni, hogy két újraépítés ne írja felül egymás
eredményét. A régi példány akkor szabadulhat fel, amikor az utolsó olvasó is
elengedte. Ennek ára a teljes újraépítés és átmenetileg két index memóriája.

Például a forrásadatban az `assoc` azonos ID-n frissít, a `dissoc` töröl;
utána `api/index` építi a következő teljes indexet. Ez ma is megvalósítható
alkalmazási szinten. Nem nevezhető helyben végzett HNSW-frissítésnek.

### Javasolt fejlesztési sorrend

1. Egységes írási szerződés, dimenzió/ID validálás és verziózott snapshot.
2. `mark-deleted!`, `unmark-deleted!`, aktív elemszám, perzisztált törlési
   állapot. A törölt csomópont keresési útvonalként használható, eredményként
   nem; mind a Clojure, mind a kompakt Java útvonalnak támogatnia kell ezt.
   A végső top-k lista utólagos szűrése önmagában nem elég.
3. Aktív vektorokból újraépítő `compact/rebuild` művelet. Ez állítja vissza
   a tárhelyet, megszünteti a régi éleket, és biztonságos alapot ad a
   frissítésekhez.
4. `upsert!`: új ID-n beszúrás, meglévőn helyi éljavítás vagy verziózott
   új csomópont. Az éljavítást a szomszédokra is ki kell terjeszteni;
   sok egymás utáni módosításon elérhetőségi és recall-teszt szükséges.
5. Nagy írási terheléshez base+delta index és háttérben összevonás.

## Újabb kutatások: mi illeszthető ide?

| Forrás | Lényeg | Döntés ehhez a repóhoz |
| --- | --- | --- |
| [Distance Comparison Operation Optimization in ANNS, EDBT 2026](https://openproceedings.org/2026/conf/edbt/paper-270.pdf) | A távolságszámítás, a SIMD, a memóriaelrendezés és a közelítő pruning együttes hatását vizsgálja; a bonyolult pruning előnye SIMD mellett csökkenhet | Elsőként a teljes távolságszámítás egyszerű, SIMD-barát float32 útját valósítottam meg; ez a cikkből levont mérnöki döntés, nem a cikk új algoritmusának portja |
| [VSAG, VLDB 2025](https://www.vldb.org/pvldb/vol18/p5017-cheng.pdf) | Memóriahozzáférés, prefetch, részben redundáns tárolás, kvantálás és szelektív újrarangsorolás | A kompakt adat és SIMD iránya releváns; a teljes prefetch/kvantálási rendszert külön kísérletként érdemes mérni |
| [FGIM, SIGMOD 2026](https://arxiv.org/abs/2603.21710) | Már létező közelségi gráfokat alakít át, javít és egyesít | Jó jelölt base+delta és nagy batch összevonásra; új építési folyamat kell hozzá, a publikált gyorsulás nem vonatkoztatható közvetlenül az üres gráfból építésünkre |
| [OPT-SNG, 2026. májusi v5 preprint](https://arxiv.org/abs/2509.15531v5) | A gráf ritkítási paraméterét elméleti szabállyal választja, csökkentve a paraméterkeresés költségét | A nagy headline gyorsulást nem szabad egyetlen fix `M`/`efConstruction` építésre ígérni; csak saját recall–idő összehasonlítás után adaptálható |
| [IP-DiskANN, 2025 preprint](https://arxiv.org/abs/2502.13826) | Beszúrás/törlés helyben, batch konszolidáció nélkül, hosszú frissítéssorozatokkal vizsgálva | Hasznos tervezési alap, de más gráfalgoritmus; nem kis HNSW-patch |
| [MN-RU, 2024](https://arxiv.org/abs/2407.07871) | HNSW-frissítések során elérhetetlenné váló pontok kezelése | Régebbi, de közvetlenül releváns figyelmeztetés az upsert/törlés és a gráf elérhetőségi tesztjeihez |

A [RaBitQ referencia-könyvtár HNSW leírása](https://vectordb-ntu.github.io/RaBitQ-Library/index/hnsw/)
külön is fontos: a gráféleket ott nyers vektorokból építik és ritkítják.
A keresési kvantálás bevezetése ezért önmagában nem jelent automatikusan
gyorsabb indexépítést. Az agresszív kvantálás vagy kevesebb jelölt bejárása
megváltoztathatja a gráfot és a recallt.

## Mi került ténylegesen a kódba?

- `:construction-precision :float32`: külön bekapcsolható, beépített cosine
  metrikához használható építési mód. Az alapérték továbbra is `:float64`.
- Csomópontonként egyszer készül float32 építési vektor; a beszúrás,
  szomszédválasztás és élritkítás ebből dolgozik.
- Java `FloatVector` kernel négy független akkumulátorral, maradékelemek
  kezelésével; Vector API nélkül skaláris Clojure út.
- A normalizált double forrásvektor megmarad, a kompakt keresési snapshot
  precízióját továbbra is külön a `:compact-precision` szabályozza.
  A kompakt kernel nélküli gráfkeresés az építési reprezentációt használja.
- Mentés/betöltés a fő API-n, beépített metrika és építési precízió
  megőrzésével. Régi, precíziómező nélküli EDN-indexek double építési
  reprezentációval tölthetők be. Régi, metrika nélküli euclidean fájlnál
  továbbra is meg kell adni az eredeti távolságfüggvényt.
- Megismételhető, felváltott sorrendű benchmark, a gráfépítés és snapshot
  idejét külön jelentve, teljes végigkeresésből számított top-10 referenciával.

Ez sebesség–memória csere: a float32 építési cache további `N × D × 4`
bájtot használ. A double forrás és a keresési snapshot mellett marad meg,
tehát az egész index memóriaigényét ez a változtatás nem felezi.
Közeli távolságoknál a float32 kerekítés megváltoztathatja az élválasztást;
a tesztek és mérések nem jelentenek minden adathalmazra recall-garanciát.

```clojure
(require '[hnsw.api :as hnsw])

(def index
  (hnsw/index vectors
              :metric :cosine
              :construction-precision :float32
              :compact-precision :float32
              :num-threads 10
              :M 16
              :ef-construction 200
              :ef-search 50))

(hnsw/save-index index "index.edn")
(def restored (hnsw/load-index "index.edn"))
```

Fordítás és JVM-paraméter: `scripts/compile_vector_kernel.sh`, majd
`clojure -J--add-modules=jdk.incubator.vector ...`.

## Mérési módszer

A `examples/bible_benchmark/run_construction.sh` a meglévő közös MPNet
vektorfájlokat használja, 10 000 és 31 126 magyar vektorral, 768 dimenzióban.
Az angol lekérdezések magyar top-10 szomszédait teljes végigkeresés adja meg.
A mért szakasz nem tartalmaz fájlbetöltést, embedding-generálást,
referenciaszámítást vagy explicit GC-t. A JVM melegítése után minden
precízióhoz friss gráf készül; a futási sorrend ismétlésenként megfordul.
A párhuzamos élbeszúrás sorrendje miatt a recall azonos seed mellett is
kissé változhat.

```bash
NUM_VECTORS=10000 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  ./examples/bible_benchmark/run_construction.sh

NUM_VECTORS=31173 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  ./examples/bible_benchmark/run_construction.sh
```

A `NUM_VECTORS=31173` felső korlát: a ténylegesen betöltött magyar korpusz
31 126 nem üres vers. A 10k mérés ennek a korpusznak az első 10 000 eleme,
nem a README korábbi szintetikus 10k adathalmaza.

## Eredmények

Apple Silicon, Java 21.0.8, 10 szál, 768 dimenzió; `M=16`,
`efConstruction=200`, `efSearch=50`, `k=10`. Öt teljes újraépítés/mód,
1000 angol lekérdezés, mediánok:

| Magyar vektorok | Építési mód | Gráfépítés | Snapshot | Teljes idő | ANN Recall@10 |
| --- | --- | --- | --- | --- | --- |
| 10 000 | float64 | 1,670 s | 0,070 s | 1,736 s | 98,52% |
| 10 000 | float32 | 1,094 s | 0,074 s | 1,169 s | 98,47% |
| 31 126 | float64 | 9,052 s | 0,237 s | 9,285 s | 97,75% |
| 31 126 | float32 | 6,079 s | 0,259 s | 6,338 s | 97,71% |

A teljes építési idő csökkenése **32,7%**, illetve **31,7%**; az építési
sebesség **1,48×**, illetve **1,46×**. A median recall eltérése −0,05,
illetve −0,04 százalékpont. Az ötfutásos recall-tartományok:

- 10k float64: 98,28–98,55%; float32: 98,40–98,51%.
- Teljes Biblia float64: 97,69–97,95%; float32: 97,70–97,81%.

A snapshot és gráfépítés mediánjának összege nem feltétlenül a teljes idő
mediánja. Egyes időminták ingadoznak; nem csak a legjobb futást közöljük.
A teljes Biblián a float32 építési cache 91,2 MiB többletet jelent, a
tömbfejléceken felül. A default float64 mód nem kap ilyen másolatot.

A nyers eredmények és a vektorfájlok SHA-256 azonosítói:
[`construction_results_2026-09-05.edn`](../examples/bible_benchmark/construction_results_2026-09-05.edn).
Ez a két Clojure építési mód összehasonlítása, nem új Python-mérés.

## Ellenőrzések

- Teljes suite, Java Vector API-val: **66 teszt, 302 assertion, 0 hiba**.
- Végső célzott SIMD regressziók: **22 teszt, 176 assertion, 0 hiba**.
- Az új funkciók Vector API nélkül: **9 teszt, 117 assertion, 0 hiba**.
- A régi, nem normalizált cosine fájl betöltését és ismételt mentését
  külön regresszió is ellenőrzi: **1 teszt, 3 assertion, 0 hiba**.
- A benchmark futtatható scriptje, EDN-eredményei és rövid smoke futása
  ellenőrizve; `git diff --check` hibamentes.

A tesztek lefedik a SIMD-ciklus maradékelemeit, zéró és előjeles
vektorokat, a teljes végigkereséshez mért recallt, a párhuzamos gráf
fokszám- és élhivatkozási invariánsait, beszúrást betöltés után,
custom metrikák explicit visszaadását, régi EDN-formátumot és a simple API-t.
A korábbi kompakt float32 teszt két külön párhuzamos gráf véletlenszerű
eltérését is mérte; most egy szálon, azonos seeddel épít, hogy a vizsgálat
csak a keresési reprezentáció eltérésére vonatkozzon.

```bash
./scripts/compile_vector_kernel.sh
clojure -J--add-modules=jdk.incubator.vector -M:test :all
```
