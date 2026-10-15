# Panduan Teknik Analisis Trading: Crypto, Forex, Saham AS, dan Indeks

> **Tujuan:** panduan belajar yang menjelaskan 20 teknik analisis, rumus utama, cara membaca hasil, contoh hitung, serta keterbatasan masing-masing.
>
> **Peringatan:** materi ini bersifat edukasi, bukan rekomendasi membeli atau menjual aset tertentu. Rumus membantu menstrukturkan analisis, tetapi tidak menjamin keuntungan. Gunakan data yang sesuai, uji strategi pada data historis, dan kelola risiko.

## Daftar isi

1. [Dasar dan notasi](#dasar-dan-notasi)
2. [Crypto](#1-crypto)
3. [Forex](#2-forex)
4. [Saham AS](#3-saham-as)
5. [Indeks pasar](#4-indeks-pasar)
6. [Rencana menggabungkan analisis](#5-cara-menggabungkan-teknik)
7. [Manajemen risiko dan pengujian](#6-manajemen-risiko-dan-pengujian)
8. [Glosarium](#7-glosarium)

---

## Dasar dan notasi

Rumus dalam panduan ini adalah bentuk umum. Penyedia data atau platform dapat memakai definisi, periode, dan metode agregasi yang sedikit berbeda. Selalu periksa dokumentasi indikator yang digunakan.

| Simbol | Arti |
|---|---|
| `O, H, L, C` | Harga pembukaan, tertinggi, terendah, dan penutupan pada suatu periode |
| `V` | Volume transaksi |
| `P_t` | Harga pada periode `t` |
| `P_(t-1)` | Harga pada periode sebelumnya |
| `MA_n` | Rata-rata bergerak selama `n` periode |
| `EMA_n` | Rata-rata bergerak eksponensial selama `n` periode |
| `N` | Jumlah observasi |
| `Δ` | Perubahan dari satu periode ke periode berikutnya |
| `ln(x)` | Logaritma natural dari `x` |

### Dua rumus dasar yang sering dipakai

**Persentase perubahan harga**

\[
\text{Return (\%)}=\frac{P_t-P_{t-1}}{P_{t-1}}\times100\%
\]

Contoh: harga naik dari 100 ke 105, return = `(105 - 100) / 100 × 100% = 5%`.

**Volume rata-rata**

\[
\overline{V}_n=\frac{1}{n}\sum_{i=1}^{n}V_i
\]

Volume saat ini yang jauh di atas rata-rata dapat menunjukkan aktivitas yang meningkat, tetapi tidak otomatis berarti harga akan naik.

---

# 1. Crypto

Pasar crypto berjalan 24/7 pada banyak bursa, dan aset yang sama bisa memiliki likuiditas berbeda antar-platform. Data on-chain hanya mencakup aktivitas yang tercatat pada blockchain terkait; data derivatif bergantung pada bursa dan kontraknya.

## 1.1 Technical Analysis (analisis teknikal)

**Apa yang dianalisis?** Grafik harga, tren, momentum, support/resistance, dan pola candle. Teknik ini digunakan untuk menyusun skenario entry, exit, dan invalidasi.

### Rumus inti

**Simple Moving Average (SMA)**

\[
SMA_n=\frac{1}{n}\sum_{i=0}^{n-1}C_{t-i}
\]

Contoh: penutupan lima hari adalah 10, 11, 12, 11, 16. `SMA_5 = (10+11+12+11+16)/5 = 12`.

**Exponential Moving Average (EMA)**

\[
EMA_t=\alpha C_t+(1-\alpha)EMA_{t-1},\quad \alpha=\frac{2}{n+1}
\]

EMA memberi bobot lebih besar pada data terbaru. Nilai awal EMA perlu ditentukan, misalnya dengan SMA awal.

**Relative Strength Index (RSI)**

\[
RSI=100-\frac{100}{1+RS},\quad RS=\frac{\text{rata-rata kenaikan}}{\text{rata-rata penurunan}}
\]

RSI biasanya memakai 14 periode dan metode smoothing Wilder. RSI di atas 70 sering disebut *overbought* dan di bawah 30 *oversold*, tetapi dalam tren kuat RSI dapat bertahan di area tersebut.

**MACD**

\[
MACD=EMA_{12}-EMA_{26}
\]
\[
Signal=EMA_9(MACD),\quad Histogram=MACD-Signal
\]

Periode 12, 26, dan 9 adalah setelan populer, bukan angka yang selalu optimal.

### Cara membaca

- Harga di atas MA yang menanjak dapat mendukung skenario tren naik.
- RSI mengukur momentum, bukan jaminan pembalikan.
- Persilangan MACD dan garis sinyal dapat membantu mengidentifikasi perubahan momentum.
- Tandai support/resistance sebagai **zona**, bukan satu harga yang pasti menahan pasar.

**Keterbatasan:** indikator umumnya tertinggal dari harga dan dapat memberi sinyal palsu saat pasar bergerak mendatar. Konfirmasi dengan struktur harga, volume, dan risiko transaksi.

## 1.2 On-Chain Analysis

**Apa yang dianalisis?** Data publik blockchain, seperti jumlah alamat aktif, transaksi, perpindahan koin, suplai, dan arus aset ke/dari bursa.

### Rumus dan metrik

**Perubahan jumlah alamat aktif**

\[
\Delta A(\%)=\frac{A_t-A_{t-1}}{A_{t-1}}\times100\%
\]

`A_t` adalah jumlah alamat aktif pada periode sekarang. Contoh: naik dari 100.000 menjadi 110.000 berarti kenaikan 10%.

**Net exchange flow**

\[
\text{Net Flow}=\text{Inflow ke bursa}-\text{Outflow dari bursa}
\]

Net flow positif berarti lebih banyak aset masuk ke alamat bursa yang dilacak daripada keluar pada periode itu. Ini *dapat* menambah potensi suplai untuk dijual, tetapi tidak membuktikan bahwa aset pasti akan dijual.

**Market Value to Realized Value (MVRV)**

\[
MVRV=\frac{\text{Market Capitalization}}{\text{Realized Capitalization}}
\]

Market cap biasanya dihitung dari harga pasar dikali suplai beredar. Realized cap menggunakan nilai harga saat koin terakhir bergerak, menurut metodologi penyedia data. MVRV tinggi/rendah perlu dibandingkan dengan sejarah aset dan metodologi yang sama.

### Cara membaca

- Pertumbuhan aktivitas jaringan bisa mengindikasikan penggunaan meningkat; periksa apakah pertumbuhan itu organik.
- Arus masuk bursa dapat menjadi sinyal yang perlu diperhatikan, bukan sinyal jual otomatis.
- Bandingkan metrik dengan harga, siklus pasar, dan data historis.

**Keterbatasan:** satu orang dapat mengendalikan banyak alamat; perpindahan internal bursa dapat mengaburkan gambaran; blockchain yang berbeda tidak selalu sebanding. Metrik on-chain juga tidak tersedia dengan cara yang sama untuk semua token.

## 1.3 Order Flow Analysis

**Apa yang dianalisis?** Transaksi yang benar-benar terjadi, agresivitas pembeli/penjual, serta likuiditas bid dan ask. Order book menunjukkan pesanan yang tersedia, bukan jaminan pesanan itu akan tetap ada.

### Rumus dan metrik

**Cumulative Volume Delta (CVD)**

\[
Delta_t=V_{\text{aggressive buy},t}-V_{\text{aggressive sell},t}
\]
\[
CVD_t=CVD_{t-1}+Delta_t
\]

Klasifikasi agresor tergantung metode bursa/vendor. Volume beli agresif biasanya berarti transaksi yang mengangkat ask; jual agresif biasanya memukul bid.

**Bid-ask spread**

\[
Spread=Ask_{terbaik}-Bid_{terbaik}
\]
\[
Spread(\text{bps})=\frac{Ask-Bid}{(Ask+Bid)/2}\times10{,}000
\]

bps = basis point; 100 bps = 1%.

**Order-book imbalance** (contoh sederhana untuk kedalaman beberapa level)

\[
Imbalance=\frac{Q_{bid}-Q_{ask}}{Q_{bid}+Q_{ask}}
\]

`Q_bid` dan `Q_ask` adalah kuantitas bid dan ask pada kedalaman yang sama. Nilai berkisar -1 sampai +1 bila keduanya tidak negatif dan jumlahnya bukan nol.

### Cara membaca

- CVD meningkat saat harga naik dapat mendukung adanya pembelian agresif.
- Harga naik tetapi CVD melemah bisa menandakan divergensi yang perlu diselidiki.
- Spread sempit biasanya berarti biaya masuk/keluar lebih rendah, tetapi spread dapat melebar cepat saat volatilitas tinggi.

**Keterbatasan:** order dapat dibatalkan, volume antar-bursa terfragmentasi, dan data agresor tidak selalu seragam. Jangan menyamakan banyaknya order yang terlihat dengan niat pasar yang pasti.

## 1.4 Derivatives Analysis

**Apa yang dianalisis?** Futures/perpetual, leverage, open interest, funding rate, dan likuidasi.

### Rumus dan metrik

**Perubahan open interest (OI)**

\[
\Delta OI(\%)=\frac{OI_t-OI_{t-1}}{OI_{t-1}}\times100\%
\]

OI adalah nilai atau jumlah kontrak yang masih terbuka, tergantung cara pelaporan. Kenaikan OI berarti kontrak terbuka bertambah, bukan berarti semua pelaku bullish.

**Funding payment perkiraan**

\[
\text{Pembayaran funding}\approx\text{Nilai posisi}\times\text{Funding Rate}
\]

Contoh: posisi nominal 10.000 USDT dan funding rate +0,01% menghasilkan pembayaran sekitar 1 USDT untuk satu interval bagi pihak yang membayar, sebelum detail aturan bursa. Pada banyak perpetual, rate positif berarti long membayar short, tetapi periksa aturan kontrak.

**Basis futures**

\[
Basis=Futures\ Price-Spot\ Price
\]
\[
Basis(\%)=\frac{Futures-Spot}{Spot}\times100\%
\]

### Cara membaca

- Harga naik + OI naik dapat berarti posisi baru masuk; arah bersih tidak bisa dipastikan dari OI saja.
- Funding sangat positif dapat menunjukkan long relatif mahal/padat; ini bukan prediksi pasti bahwa harga akan turun.
- Basis perlu ditafsirkan sesuai jatuh tempo kontrak, biaya pendanaan, dan kondisi pasar.

**Keterbatasan:** data berbeda antar-bursa; leverage dapat memicu likuidasi berantai; OI tidak mengungkap arah masing-masing posisi secara lengkap. Derivatif dapat menyebabkan kerugian melebihi perkiraan, terutama jika aturan margin tidak dipahami.

## 1.5 Fundamental Analysis Crypto

**Apa yang dianalisis?** Kegunaan jaringan, pengguna, keamanan, tokenomics, tata kelola, pendapatan/protokol, dan valuasi relatif.

### Rumus dan metrik

**Market capitalization**

\[
Market\ Cap=Harga\ Token\times Suplai\ Beredar
\]

**Fully Diluted Valuation (FDV)**

\[
FDV=Harga\ Token\times Suplai\ Maksimum\ atau\ Suplai\ Terdilusi\ Penuh
\]

Pastikan definisi suplai yang dipakai sumber data. FDV tinggi relatif terhadap market cap dapat menunjukkan potensi dilusi ketika token baru beredar, tetapi dampaknya bergantung pada jadwal unlock dan permintaan.

**Price-to-sales atau price-to-fees (hanya bila definisi metrik sesuai)**

\[
Multiple=\frac{\text{Valuasi yang dipilih}}{\text{Pendapatan/fees selama periode yang konsisten}}
\]

Untuk protokol, “fees”, “revenue”, dan pendapatan yang benar-benar diterima pemegang token bukan hal yang sama. Gunakan definisi yang konsisten.

### Checklist kualitatif

- Apakah token benar-benar diperlukan untuk penggunaan jaringan?
- Bagaimana distribusi token, unlock, emisi, dan konsentrasi kepemilikan?
- Apakah jaringan aman dan cukup terdesentralisasi untuk tujuan yang diklaim?
- Apakah pengguna, transaksi, atau pendapatan tumbuh secara berkelanjutan?
- Apa risiko regulasi, teknis, kompetisi, dan tata kelola?

**Keterbatasan:** banyak proyek belum memiliki arus kas yang bisa dinilai seperti perusahaan; metrik dapat dimanipulasi atau disalahartikan; kualitas proyek tidak menjamin tokennya naik.

---

# 2. Forex

Forex memperdagangkan pasangan mata uang. Harga EUR/USD, misalnya, menunjukkan berapa USD yang diperlukan untuk membeli satu EUR. Perhatikan spread, sesi pasar, biaya swap/rollover, ukuran kontrak, dan jadwal rilis ekonomi.

## 2.1 Technical Analysis

**Apa yang dianalisis?** Struktur harga, tren, momentum, volatilitas, dan level teknikal. Rumus SMA, EMA, RSI, dan MACD sama seperti bagian crypto; yang berubah adalah instrumen, jam pasar, kualitas data, dan pengaturan periode.

### Rumus tambahan: Average True Range (ATR)

True Range pada periode `t`:

\[
TR_t=\max(H_t-L_t,\ |H_t-C_{t-1}|,\ |L_t-C_{t-1}|)
\]

ATR adalah rata-rata/smoothing dari `TR` selama `n` periode; ATR 14 sering digunakan.

### Cara membaca

- ATR tinggi berarti rentang pergerakan relatif besar, bukan arah naik/turun.
- Level support/resistance dan pola harga membantu membuat skenario.
- Ukuran stop dapat mempertimbangkan ATR, tetapi harus disesuaikan dengan strategi dan biaya.

**Keterbatasan:** candle forex dapat berbeda sedikit antar-broker karena pasar spot terdesentralisasi. Indikator tidak memprediksi berita mendadak.

## 2.2 Fundamental Analysis

**Apa yang dianalisis?** Inflasi, pertumbuhan, tenaga kerja, kebijakan bank sentral, dan kondisi eksternal.

### Rumus dan perbandingan

**Inflasi tahunan dari indeks harga**

\[
Inflasi\ YoY(\%)=\frac{CPI_t-CPI_{t-12}}{CPI_{t-12}}\times100\%
\]

Untuk data bulanan, `t-12` berarti 12 bulan sebelumnya. Rumus harus mengikuti frekuensi dan definisi seri data resmi.

**Perubahan data ekonomi terhadap ekspektasi**

\[
Surprise=Data\ Aktual-Konsensus
\]

Nilai surprise positif tidak selalu menguatkan mata uang: dampaknya bergantung pada jenis data, ekspektasi pasar, dan respons bank sentral.

**Suku bunga riil sederhana**

\[
Suku\ Bunga\ Riil\ perkiraan\approx Suku\ Bunga\ Nominal-Inflasi\ yang\ diharapkan
\]

Ini pendekatan sederhana; pengukuran yang lebih tepat memakai ekspektasi inflasi untuk horizon yang sebanding.

### Cara membaca

- Bandingkan ekonomi dan kebijakan kedua negara, bukan satu negara secara terpisah.
- Perhatikan apakah data sudah diantisipasi pasar.
- Gunakan kalender ekonomi untuk mengidentifikasi risiko lonjakan spread dan volatilitas.

**Keterbatasan:** rilis ekonomi dapat direvisi; pasar sering bergerak berdasarkan selisih dari ekspektasi, bukan sekadar angka aktual.

## 2.3 Currency Strength Analysis

**Apa yang dianalisis?** Kinerja relatif satu mata uang terhadap sekumpulan mata uang lain. Tidak ada satu rumus universal; skor strength bergantung pada pasangan, bobot, dan periode yang dipilih.

### Contoh rumus berbasis return

Untuk mata uang `A`, buat daftar pasangan yang melibatkan A. Definisikan return dengan tanda positif jika A menguat dan negatif jika A melemah:

\[
Strength_A=\frac{\sum_{j=1}^{N}w_jr_{A,j}}{\sum_{j=1}^{N}w_j}
\]

`r_(A,j)` adalah return yang sudah diseragamkan arahnya; `w_j` adalah bobot (misalnya sama rata). Jangan langsung menjumlahkan return mentah pasangan tanpa membalik tanda ketika mata uang berada di sisi kuotasi.

**Contoh konseptual:** jika skor USD positif terhadap banyak mata uang dan skor EUR negatif, EUR/USD mungkin mendapat tekanan. Itu hanya hipotesis relatif, bukan sinyal entry otomatis.

### Langkah praktis

1. Pilih horizon konsisten, misalnya 1 hari atau 1 minggu.
2. Hitung return tiap pasangan.
3. Samakan tanda berdasarkan apakah mata uang sedang dinilai sebagai base atau quote.
4. Gunakan bobot yang konsisten dan bandingkan hasil lintas mata uang.
5. Konfirmasi dengan grafik pasangan yang hendak diperdagangkan.

**Keterbatasan:** hasil berubah menurut pasangan, periode, dan bobot; mata uang yang “kuat” tidak berarti semua pasangannya naik.

## 2.4 Sentiment & Positioning Analysis

**Apa yang dianalisis?** Posisi pelaku tertentu, sentimen trader ritel, dan data positioning. Di pasar futures mata uang, laporan CFTC Commitment of Traders (COT) sering dipakai, tetapi bukan data real-time.

### Rumus COT sederhana

\[
Net\ Position=Long\ Contracts-Short\ Contracts
\]
\[
Net\ Position\ sebagai\ \%\ OI=\frac{Long-Short}{Open\ Interest}\times100\%
\]

Pastikan kategori pelaku dan kontrak yang dibandingkan sama pada setiap tanggal. Laporan COT diterbitkan berkala dan memiliki jeda waktu.

**Sentimen ritel:** jika penyedia melaporkan persentase long dan short:

\[
Long\ Share=\frac{Jumlah\ akun\ long}{Jumlah\ akun\ long+short}\times100\%
\]

Angka akun tidak selalu sama dengan besaran uang atau ukuran posisi.

### Cara membaca

- Posisi ekstrem dapat menandakan konsentrasi, tetapi bisa bertahan lama.
- Perubahan positioning dari minggu ke minggu sering lebih informatif daripada satu angka tunggal.
- Gunakan sebagai konteks, bukan alasan tunggal untuk melawan tren.

**Keterbatasan:** COT hanya mencakup pasar dan kategori yang dilaporkan; tidak mewakili seluruh pasar spot forex. Sentimen dari satu broker juga tidak mewakili semua trader.

## 2.5 Interest Rate Differential Analysis

**Apa yang dianalisis?** Perbedaan suku bunga dan ekspektasi kebijakan dua negara, yang dapat memengaruhi arus modal dan nilai tukar.

### Rumus

\[
Differential=i_A-i_B
\]

`i_A` dan `i_B` adalah suku bunga acuan atau yield dengan tenor yang sebanding. Contoh: jika yield negara A 4,5% dan negara B 3,0%, differential = +1,5 poin persentase untuk A.

**Perubahan differential**

\[
\Delta Differential=(i_{A,t}-i_{B,t})-(i_{A,t-1}-i_{B,t-1})
\]

Bandingkan tenor yang sama—misalnya yield obligasi 2 tahun kedua negara—dan tentukan apakah memakai suku bunga kebijakan atau yield pasar.

### Cara membaca

- Pelebaran differential ke arah A dapat mendukung A dalam kondisi tertentu.
- Ekspektasi suku bunga masa depan sering lebih penting daripada tingkat saat ini.
- Perhatikan inflasi, risiko negara, arus aman, dan apakah perubahan sudah tercermin dalam harga.

**Keterbatasan:** selisih suku bunga tidak menentukan kurs sendirian. Risiko, pertumbuhan, dan ekspektasi bisa mengalahkan efek carry.

---

# 3. Saham AS

Untuk saham individual, data laporan keuangan, struktur saham, dan kalender earnings sangat penting. Perhatikan perbedaan GAAP/non-GAAP, pemecahan saham (stock split), buyback, serta perubahan jumlah saham.

## 3.1 Technical Analysis

**Apa yang dianalisis?** Harga dan volume untuk mengidentifikasi tren, momentum, level penting, dan volatilitas. Rumus SMA, EMA, RSI, MACD, dan ATR di bagian sebelumnya juga berlaku pada saham.

### Rumus volume relatif

\[
Relative\ Volume=\frac{Volume\ saat\ ini}{Rata-rata\ volume\ untuk\ periode\ pembanding}
\]

Bandingkan waktu yang sebanding. Volume pada 10 menit pertama sesi tidak adil dibandingkan volume sehari penuh.

### Cara membaca

- Breakout dengan volume relatif tinggi dapat lebih meyakinkan daripada breakout dengan volume tipis, tetapi tetap bisa gagal.
- Sesuaikan data terhadap stock split dan perhatikan likuiditas.
- Indikator harian tidak selalu cocok untuk keputusan intraday.

**Keterbatasan:** harga dapat gap setelah earnings atau berita ketika pasar tutup. Stop order tidak menjamin harga eksekusi tertentu.

## 3.2 Fundamental Analysis

**Apa yang dianalisis?** Kemampuan perusahaan menghasilkan laba, kas, dan pertumbuhan; neraca; kualitas manajemen; serta valuasi.

### Rumus utama

**Earnings per Share (EPS)**

\[
EPS=\frac{Laba\ bersih-Dividen\ preferen}{Rata\text{-}rata\ tertimbang\ saham\ biasa}
\]

Gunakan EPS diluted jika ingin memasukkan potensi dilusi sesuai laporan perusahaan.

**Price-to-Earnings (P/E)**

\[
P/E=\frac{Harga\ per\ saham}{EPS\ tahunan}
\]

P/E kurang bermakna jika EPS negatif atau sangat tidak normal. Pastikan memakai trailing atau forward P/E secara jelas.

**Free Cash Flow (FCF) sederhana**

\[
FCF=Arus\ Kas\ Operasi-Capital\ Expenditures
\]

**Return on Equity (ROE)**

\[
ROE=\frac{Laba\ bersih}{Rata\text{-}rata\ ekuitas}\times100\%
\]

### Cara membaca

- Bandingkan valuasi dengan perusahaan sejenis dan sejarah perusahaan.
- Periksa pertumbuhan pendapatan bersama margin dan arus kas.
- Laba akuntansi yang meningkat tidak selalu berarti arus kas membaik.

**Keterbatasan:** akuntansi, siklus industri, dan item satu kali dapat mengubah rasio. Rasio tinggi tidak otomatis mahal dan rasio rendah tidak otomatis murah.

## 3.3 Earnings Analysis

**Apa yang dianalisis?** Hasil kuartalan dibandingkan dengan konsensus, panduan manajemen, dan ekspektasi investor.

### Rumus

**Earnings surprise**

\[
EPS\ Surprise(\%)=\frac{EPS_{aktual}-EPS_{konsensus}}{|EPS_{konsensus}|}\times100\%
\]

Rumus persentase ini bermasalah jika konsensus nol atau sangat dekat nol; dalam kasus itu laporkan selisih absolut. Jika EPS konsensus negatif, interpretasi persentase perlu kehati-hatian.

**Revenue growth**

\[
Revenue\ Growth(\%)=\frac{Revenue_t-Revenue_{t-1}}{Revenue_{t-1}}\times100\%
\]

Untuk pertumbuhan tahunan gunakan kuartal yang sama tahun lalu; untuk pertumbuhan kuartalan, perhatikan musim.

**Margin operasi**

\[
Operating\ Margin=\frac{Operating\ Income}{Revenue}\times100\%
\]

### Cara membaca

- Periksa EPS dan pendapatan, tetapi juga guidance, margin, arus kas, dan komentar manajemen.
- “Beat” konsensus tidak menjamin saham naik; pasar bereaksi terhadap keseluruhan hasil relatif terhadap ekspektasi.
- Catat reaksi harga setelah laporan dan volatilitasnya.

**Keterbatasan:** konsensus berbeda antar-penyedia dan dapat berubah; perusahaan dapat melaporkan angka GAAP dan non-GAAP yang berbeda.

## 3.4 Sector & Relative Strength Analysis

**Apa yang dianalisis?** Apakah saham mengungguli atau tertinggal dari benchmark atau sektor. Gunakan periode yang sama dan harga yang konsisten (idealnya total return jika membandingkan kinerja investasi).

### Rumus

**Relative performance sederhana**

\[
Relative\ Return=Return_{saham}-Return_{benchmark}
\]

**Rasio relatif**

\[
Relative\ Strength\ Ratio=\frac{Harga\ saham}{Harga\ benchmark}
\]

Jika rasio naik, saham mengungguli benchmark dalam periode tersebut secara relatif, dengan catatan kedua seri data konsisten.

**Return periode**

\[
Return(\%)=\left(\frac{P_t}{P_0}-1\right)\times100\%
\]

### Cara membaca

- Bandingkan saham dengan benchmark relevan seperti S&P 500 atau ETF sektor.
- Gunakan beberapa horizon (misalnya 1 bulan, 3 bulan, 1 tahun) agar tidak terpaku pada satu periode.
- Relative strength bukan RSI; istilahnya berbeda walaupun sama-sama mengandung kata “strength”.

**Keterbatasan:** hasil dipengaruhi pemilihan benchmark, dividen, mata uang, dan periode. Kinerja relatif yang kuat tidak menjamin kinerja masa depan.

## 3.5 Institutional & Insider Analysis

**Apa yang dianalisis?** Kepemilikan institusi dan transaksi insider yang dilaporkan secara publik. Di AS, formulir SEC seperti 13F dan Form 4 sering menjadi sumber; jenis laporan dan tenggat pelaporannya berbeda.

### Rumus dan metrik

**Perubahan jumlah saham institusi yang dilaporkan**

\[
Change(\%)=\frac{Shares_t-Shares_{t-1}}{Shares_{t-1}}\times100\%
\]

Bandingkan periode laporan yang sebanding dan periksa apakah jumlah saham disesuaikan untuk split atau perubahan korporasi.

**Persentase kepemilikan**

\[
Ownership(\%)=\frac{Saham\ yang\ dimiliki}{Total\ saham\ beredar}\times100\%
\]

Pastikan tanggal kepemilikan dan total saham beredar sesuai.

### Cara membaca

- Laporan institusi sering tertinggal dari kondisi saat ini.
- Bedakan transaksi insider yang benar-benar pembelian/penjualan di pasar dari pemberian saham, opsi, atau pemotongan pajak.
- Lihat pola dan konteks, bukan satu transaksi terisolasi.

**Keterbatasan:** 13F bukan gambaran lengkap semua posisi dan umumnya tidak menunjukkan transaksi real-time; posisi short dan derivatif dapat tidak terlihat dengan cara yang sama.

---

# 4. Indeks Pasar

Indeks seperti S&P 500, Nasdaq-100, dan Dow Jones mengukur kelompok saham dengan metode pembobotan berbeda. Indeks itu sendiri tidak selalu dapat dibeli langsung; produk yang diperdagangkan bisa berupa ETF, futures, opsi, atau CFD, masing-masing dengan karakteristik dan biaya berbeda.

## 4.1 Technical Analysis

**Apa yang dianalisis?** Tren indeks, momentum, level harga, dan volatilitas. Rumus SMA, EMA, RSI, MACD, dan ATR tetap berlaku.

### Rumus tambahan: drawdown

\[
Drawdown_t=\frac{P_t-Peak_t}{Peak_t}\times100\%
\]

`Peak_t` adalah harga puncak sebelumnya atau puncak tertinggi hingga waktu `t`, sesuai definisi yang dipilih. Drawdown biasanya nol saat harga menyentuh puncak baru dan negatif setelah turun.

**Maximum drawdown**

\[
MDD=\min(Drawdown_t)
\]

### Cara membaca

- Indeks di atas MA yang naik dapat mendukung tren naik.
- Drawdown menunjukkan penurunan dari puncak, bukan besarnya kerugian semua investor.
- Untuk futures/CFD, perhatikan jam perdagangan, gap, margin, dan spesifikasi kontrak.

**Keterbatasan:** indeks dapat tampak kuat meskipun hanya beberapa konstituen yang naik. Gunakan market breadth sebagai pelengkap.

## 4.2 Market Breadth Analysis

**Apa yang dianalisis?** Seberapa banyak saham yang ikut naik atau turun di balik pergerakan indeks.

### Rumus

**Advance-Decline (A-D) difference**

\[
AD\ Difference_t=Advancers_t-Decliners_t
\]

**Advance-Decline Line**

\[
ADL_t=ADL_{t-1}+AD\ Difference_t
\]

**Persentase saham di atas moving average**

\[
\%\ Above\ MA_n=\frac{Jumlah\ saham\ di\ atas\ MA_n}{Jumlah\ saham\ yang\ dihitung}\times100\%
\]

**Advance/Decline ratio**

\[
A/D\ Ratio=\frac{Advancers}{Decliners}
\]

Rasio tidak terdefinisi jika jumlah decliners = 0; nyatakan kondisi itu, jangan membaginya dengan nol.

### Cara membaca

- Indeks naik dan breadth meluas dapat menunjukkan partisipasi yang lebih luas.
- Indeks naik sementara breadth melemah dapat menjadi tanda konsentrasi kenaikan.
- Tentukan semesta saham secara konsisten dan tangani saham yang tidak memiliki data.

**Keterbatasan:** saham dengan bobot besar dapat mendominasi indeks berbobot kapitalisasi walau mayoritas saham turun. Breadth bergantung pada definisi universe dan data.

## 4.3 Sector Rotation Analysis

**Apa yang dianalisis?** Pergeseran kinerja relatif antar sektor, misalnya teknologi, kesehatan, energi, atau barang konsumsi. Rotasi dapat dipengaruhi siklus ekonomi, suku bunga, valuasi, dan perubahan ekspektasi.

### Rumus

**Relative sector return**

\[
Relative\ Return_{sektor}=Return_{sektor}-Return_{benchmark}
\]

**Rasio relatif sektor**

\[
Sector\ Ratio=\frac{Harga\ indeks/ETF\ sektor}{Harga\ benchmark}
\]

Rasio meningkat berarti sektor mengungguli benchmark secara relatif selama periode tersebut, dengan data yang sebanding.

### Cara membaca

1. Pilih benchmark dan periode yang sama untuk semua sektor.
2. Hitung return total jika dividen tersedia.
3. Urutkan kinerja relatif, lalu periksa apakah tren bertahan di beberapa horizon.
4. Hubungkan dengan konteks ekonomi, tetapi jangan menganggap pola rotasi selalu berulang.

**Keterbatasan:** klasifikasi sektor dan bobot berubah; ETF sektor memiliki biaya dan tracking error; rotasi historis tidak menjamin siklus berikutnya.

## 4.4 Macro & Interest Rate Analysis

**Apa yang dianalisis?** Inflasi, tenaga kerja, pertumbuhan ekonomi, kebijakan bank sentral, dan yield obligasi yang memengaruhi valuasi saham dan ekspektasi ekonomi.

### Rumus yang berguna

**Perubahan yield**

\[
\Delta Yield=Yield_t-Yield_{t-1}
\]

Yield sering dikutip dalam poin persentase atau basis point. Contoh: 4,20% menjadi 4,35% = naik 0,15 poin persentase = 15 basis point.

**Inflasi tahunan**

\[
Inflasi\ YoY(\%)=\frac{CPI_t-CPI_{t-12}}{CPI_{t-12}}\times100\%
\]

**Kejutan data**

\[
Surprise=Aktual-Konsensus
\]

### Cara membaca

- Kenaikan yield dapat menekan valuasi saham berdurasi panjang dalam sebagian kondisi, tetapi tidak selalu.
- Pisahkan perubahan yield nominal, ekspektasi inflasi, dan yield riil bila datanya tersedia.
- Catat jadwal FOMC, CPI, data tenaga kerja, dan rilis penting lainnya.

**Keterbatasan:** hubungan makro-pasar berubah menurut rezim. Satu rilis data tidak cukup untuk menetapkan arah indeks.

## 4.5 Volatility & Options Analysis

**Apa yang dianalisis?** Ekspektasi volatilitas dari opsi, harga perlindungan downside, dan posisi opsi. VIX mengukur volatilitas tersirat 30 hari yang diturunkan dari opsi indeks S&P 500 menurut metodologi Cboe; VIX bukan probabilitas langsung bahwa pasar akan turun.

### Rumus dan metrik

**Perubahan VIX**

\[
\Delta VIX(\%)=\frac{VIX_t-VIX_{t-1}}{VIX_{t-1}}\times100\%
\]

**Put/call volume ratio sederhana**

\[
Put/Call\ Ratio=\frac{Volume\ opsi\ put}{Volume\ opsi\ call}
\]

Pastikan membandingkan jenis volume yang sama (misalnya volume harian atau open interest), cakupan kontrak, dan metodologi penyedia.

**Perkiraan gerak satu simpangan baku dari volatilitas tahunan**

\[
Expected\ Move\approx P\times\sigma_{annual}\times\sqrt{\frac{d}{365}}
\]

`P` = level indeks, `σ_annual` = volatilitas tahunan dalam bentuk desimal, `d` = jumlah hari kalender. Ini pendekatan kasar dengan asumsi skala akar waktu; kalender perdagangan dan metodologi opsi dapat berbeda.

### Cara membaca

- VIX naik biasanya menunjukkan opsi S&P 500 mengindikasikan volatilitas yang lebih tinggi.
- Put/call ratio tinggi bisa berkaitan dengan permintaan put yang besar, tetapi dapat pula berasal dari strategi spread atau lindung nilai.
- Volatilitas mengukur besarnya gerak yang diperkirakan, bukan arah geraknya.

**Keterbatasan:** opsi memiliki expiry, strike, dan strategi yang berbeda. VIX dapat berubah cepat dan tidak memberi sinyal arah yang pasti.

---

# 5. Cara Menggabungkan Teknik

Gunakan teknik yang berbeda untuk menjawab pertanyaan yang berbeda, bukan menumpuk indikator yang mengukur hal serupa.

| Pertanyaan | Teknik yang membantu |
|---|---|
| Ke mana tren dan momentum bergerak? | Technical analysis |
| Apakah aktivitas mendukung pergerakan? | Volume, order flow, market breadth |
| Apa yang mungkin mendorong harga secara fundamental? | Fundamental, earnings, on-chain, makro |
| Apakah positioning terlalu padat atau biaya leverage berubah? | OI, funding, COT, opsi |
| Apakah aset mengungguli benchmark? | Relative strength dan sector analysis |
| Berapa risiko dan ukuran posisi yang masuk akal? | ATR/volatilitas dan manajemen risiko |

### Contoh kerangka keputusan (bukan sinyal trading)

1. **Konteks:** identifikasi tren timeframe lebih tinggi dan kalender berita.
2. **Hipotesis:** tuliskan kondisi yang mendukung skenario naik/turun.
3. **Konfirmasi:** cari bukti independen, misalnya struktur harga dan breadth; jangan menganggap semua indikator independen.
4. **Invalidasi:** tentukan kondisi yang membatalkan hipotesis sebelum masuk.
5. **Risiko:** hitung ukuran posisi berdasarkan jarak stop dan batas kerugian.
6. **Catatan:** rekam alasan, hasil, biaya, dan apakah aturan dipatuhi.

---

# 6. Manajemen Risiko dan Pengujian

## 6.1 Ukuran posisi berdasarkan risiko

\[
Risk\ Budget=Equity\times Risk\ Percentage
\]

\[
Position\ Size\approx\frac{Risk\ Budget}{Kerugian\ per\ unit\ jika\ stop\ terpicu}
\]

Contoh sederhana: ekuitas 10.000 unit mata uang akun, batas risiko 0,5% → anggaran risiko 50. Jika jarak entry ke stop berarti potensi rugi 2 per unit, ukuran posisi teoritis = 50/2 = 25 unit. Ini belum memasukkan biaya, slippage, gap, nilai kontrak, atau konversi mata uang; sesuaikan dengan spesifikasi instrumen.

Untuk forex, hitung nilai pip berdasarkan pasangan, ukuran lot, dan mata uang akun. Untuk futures/CFD, periksa nilai tick/point dan aturan margin.

## 6.2 Rasio reward-to-risk

\[
Reward/Risk=\frac{|Target-Entry|}{|Entry-Stop|}
\]

Rasio 2 berarti target berjarak dua kali jarak stop. Rasio ini tidak menunjukkan probabilitas target tercapai.

## 6.3 Expectancy strategi

\[
Expectancy=(p_w\times AvgWin)-(p_l\times AvgLoss)
\]

`p_w` = proporsi trade menang, `p_l` = proporsi trade kalah, dan `p_w+p_l=1` bila hanya ada dua kategori. Gunakan hasil bersih setelah biaya. Strategi dengan win rate tinggi masih dapat merugi jika rata-rata kerugiannya jauh lebih besar.

## 6.4 Prosedur pengujian yang layak

- Tentukan aturan entry, exit, stop, ukuran posisi, dan biaya sebelum melihat hasil.
- Hindari *look-ahead bias*: jangan memakai informasi yang belum tersedia pada saat keputusan.
- Hindari *survivorship bias*: jangan hanya menguji saham yang masih bertahan saat ini.
- Pisahkan data pengembangan dan data uji; lakukan *walk-forward testing* jika sesuai.
- Masukkan komisi, spread, funding/swap, slippage, pajak, dan keterbatasan likuiditas.
- Catat drawdown, expectancy, profit factor, jumlah trade, dan hasil di kondisi pasar berbeda.
- Uji sensitivitas parameter; hasil yang hanya bagus pada satu setelan dapat merupakan overfitting.
- Mulai dengan paper trading atau ukuran kecil sebelum mempertimbangkan risiko modal nyata.

---

# 7. Glosarium

- **ATR:** ukuran rentang pergerakan yang memasukkan gap terhadap penutupan sebelumnya.
- **Basis:** selisih harga futures dan spot.
- **CVD:** akumulasi selisih volume beli agresif dan jual agresif.
- **Drawdown:** penurunan dari puncak ekuitas atau harga sesuai definisi.
- **FDV:** valuasi token jika suplai terdilusi penuh dikalikan harga saat ini.
- **Funding rate:** pembayaran periodik antara pihak posisi perpetual menurut aturan kontrak.
- **Implied volatility:** volatilitas yang tersirat dari harga opsi berdasarkan model tertentu.
- **Market breadth:** ukuran partisipasi saham yang naik/turun.
- **OI (open interest):** kontrak derivatif yang masih terbuka.
- **Pip:** satuan perubahan harga pada forex; ukuran tepatnya bergantung pada pasangan dan kuotasi.
- **Relative strength:** kinerja suatu aset relatif terhadap benchmark; bukan indikator RSI.
- **Slippage:** perbedaan harga yang diharapkan dan harga eksekusi.
- **Spread:** selisih harga ask dan bid.
- **Yield:** tingkat imbal hasil obligasi atau instrumen pendapatan tetap.
- **bps (basis point):** 1 bps = 0,01 poin persentase; 100 bps = 1 poin persentase.

## Catatan akhir

Tidak ada satu teknik yang selalu unggul untuk semua instrumen dan kondisi pasar. Rumus sebaiknya dipakai untuk mengukur sesuatu secara konsisten, lalu ditafsirkan dalam konteks. Data, definisi, timeframe, biaya, dan disiplin risiko sama pentingnya dengan indikator itu sendiri.
