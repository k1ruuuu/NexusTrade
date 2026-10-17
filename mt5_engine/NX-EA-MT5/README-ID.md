# NX EA MT5 — sumber MQL5

Status: kode mentah versi 1.00, ditinjau secara statis. **Tidak dikompilasi**, tidak dijalankan di terminal MT5, tidak di-backtest, dan tidak melakukan transaksi. Tidak ada file EX5 dalam paket.

Paket berisi empat EA, bukan indikator dan bukan skrip Python untuk trading. Tidak ada DLL, koneksi HTTP, kredensial, martingale, grid, atau library trading eksternal. `Trade.mqh` adalah library standar bawaan MT5.

## File yang digunakan

- `Standalone/NXBreak.mq5`: satu sumber lengkap NXBreak.
- `Standalone/NXParsial.mq5`: satu sumber lengkap NXParsial.
- `Standalone/NXReverse.mq5`: satu sumber lengkap NXReverse.
- `Standalone/NX-POC.mq5`: satu sumber lengkap NX-POC.
- `Experts/*.mq5` dan `Include/*.mqh`: versi modular dengan logika yang sama, untuk pemeliharaan.
- `FLOW-AND-REVIEW.md`: spesifikasi alur, kasus kegagalan, dan batas verifikasi.

Pilih versi standalone **atau** modular. Jangan memasang dua versi strategi yang sama secara bersamaan. Versi standalone hanya memerlukan file MQ5 tersebut dan library standar MT5; versi modular memerlukan seluruh struktur folder agar include tetap tersambung.

## Aturan umum

1. Simbol mengikuti chart. Timeframe default mengikuti chart, dapat diganti melalui `InpTimeframe`.
2. Setelah dipasang/restart, EA menunggu pembukaan candle berikutnya. Tidak mengejar sinyal yang sudah lewat.
3. Hanya satu evaluasi pada candle baru; candle sinyal adalah `rates[1]`, sudah selesai. Candle berjalan hanya dipakai sebagai penanda waktu.
4. ATR memakai shift 2, sehingga candle pemicu tidak membesarkan pembanding volatilitasnya sendiri.
5. Sebelum entry: periksa data, filter sinyal, sesi, usia quote, spread, arah perdagangan, harga, SL/TP, ukuran lot, margin, dan `OrderCheck`.
6. Tidak membuka siklus jika terdapat posisi atau order apa pun pada simbol yang sama, termasuk milik manual/EA lain. NXParsial sendiri membuka tiga posisi dalam satu siklus.
7. Entry hanya dalam 30 detik pertama candle baru, termasuk sesudah pengambilan histori tick. Data lambat menyebabkan sinyal dilewati.
8. Tidak ada trailing stop, pemindahan SL ke BE, pelebaran SL, atau modifikasi TP otomatis. SL/TP dikirim bersama permintaan entry.

Waktu sesi menggunakan **waktu server broker**, bukan otomatis WIB. `0–24` berarti sepanjang hari. Untuk sesi melewati tengah malam, misalnya 22–6, jam mulai lebih besar daripada jam akhir.

## NXBreak

Buy memerlukan candle 1 bullish, open tidak di atas high candle 2, dan close melewati high candle 2 ditambah buffer. Sell adalah kebalikannya.

Buffer default 0,05 ATR. Body minimal 55% range; range candle antara 0,4 dan 2,5 ATR. Wick yang sempat menembus lalu close kembali ke dalam rentang tidak dianggap breakout. Gap yang sudah membuka di luar level juga tidak memenuhi aturan crossing.

Entry menggunakan Ask untuk buy dan Bid untuk sell. Pada saat pengiriman, Bid buy harus tetap di atas level breakout; Ask sell harus tetap di bawahnya. Jarak entry dari close sinyal maksimal 0,25 ATR, termasuk spread. Batas ini dapat membuat sinyal sah tetap dilewati.

SL default 300 **points**, dibulatkan ke kelipatan tick ke atas. TP direncanakan sebesar jarak SL, RR harga 1:1. Nilai points bukan pips: pada simbol forex 5 digit, 300 points umumnya setara 30 pips. Jangan menerapkan konversi itu ke emas/indeks tanpa membaca spesifikasi simbol.

## NXParsial

Pemicu dan filter mengikuti NXBreak. Dalam satu evaluasi, EA mengirim tiga market order berurutan, dengan lot sama, SL harga sama, dan target 1R, 3R, 6R dari harga rencana awal.

**Tiga order tidak bisa dijamin atomik atau memiliki harga fill identik.** EA memeriksa ulang quote, risiko, stop, dan margin sebelum masing-masing pengiriman. Akun harus hedging; simbol harus mendukung full-fill FOK. Netting dan simbol yang hanya mendukung IOC ditolak untuk strategi ini.

Total risiko dibagi ke tiga posisi. Jika lot minimum membuat pembagian tidak mungkin, entry dilewati; lot tidak dinaikkan agar memenuhi minimum.

Posisi ditutup oleh TP server masing-masing. Ini adalah realisasi parsial basket melalui tiga posisi, bukan tiga kali pengurangan satu posisi netting. Tidak ada pemindahan SL sisa posisi setelah TP1/TP2.

Dengan tiga lot sama, relatif terhadap risiko awal total basket:

| Hasil tiga leg | Hasil kotor dalam R basket |
|---|---:|
| SL, SL, SL | -1 |
| TP1, SL, SL | -0,3333 |
| TP1, TP2, SL | +1 |
| TP1, TP2, TP3 | +3,3333 |

Tabel menggunakan harga fill rencana identik, tanpa biaya dan slippage. Membawa seluruh volume ke TP 6R merupakan strategi exit lain; mode tersebut tidak diimplementasikan.

## NXReverse

Buy memerlukan:

- EMA20 > EMA50 > EMA200 pada candle 1.
- EMA20 dan EMA50 naik dibanding candle 2; EMA200 tidak turun.
- Candle 1 memenuhi body/range filter dan menutup di atas resistance 20 candle sebelumnya ditambah buffer.
- Open candle 1 tidak sudah berada di atas resistance tersebut.

Sell menggunakan kondisi terbalik. SNR dihitung dari high/low candle **2 sampai 21**; candle pemicu tidak masuk rentang pembanding. Ini adalah SNR rolling, bukan pivot yang menggunakan candle masa depan.

SL buy berada di bawah low terendah 10 candle selesai, dikurangi buffer 0,15 ATR. SL sell di atas high tertinggi ditambah buffer. Jarak SL harus 0,5–3 ATR; jika tidak, entry ditolak, bukan dipaksa memindahkan stop. TP direncanakan 5 kali jarak SL.

Switching recovery:

1. Posisi normal harus sudah benar-benar flat. Volume entry dan exit dalam histori posisi diperiksa agar tidak menganggap partial close sebagai penutupan penuh.
2. Hanya exit beralasan `DEAL_REASON_SL` yang mengaktifkan recovery. TP, close manual, stop-out, atau alasan lain tidak mengaktifkannya.
3. EA menunggu sinyal **berlawanan** yang memenuhi seluruh EMA/SNR/filter yang sama. Candle konfirmasi harus mulai sesudah waktu SL.
4. Kesempatan berlaku maksimal 10 bar. Selama masih menunggu, sinyal searah posisi lama dilewati. Setelah kedaluwarsa, strategi normal boleh berjalan lagi.
5. Recovery memiliki SL/TP baru dengan RR rencana 1:5. Tidak menambah lot untuk mengejar kerugian. Persentase risiko tetap dihitung dari equity saat entry, sehingga nilai uang risikonya dapat berubah.
6. Recovery yang terkena SL tidak membuat recovery kedua. Siklus berikutnya kembali membutuhkan sinyal normal.

Status recovery dan penanda exit disimpan di Global Variables terminal. Restart tidak sengaja mengubah recovery menjadi rantai martingale. Kesempatan recovery dapat dikonsumsi ketika upaya eksekusi sudah dimulai tetapi kemudian ditolak; perilaku konservatif ini menghindari pengulangan order.

## NX-POC

Profil menggunakan candle 2 sampai 101; candle sinyal tidak memengaruhi target. Histogram memiliki 40 bin secara default. Titik target adalah pusat bin terbesar, kemudian disesuaikan ke tick size.

Mode default `NX_TICK_VOLUME` menghitung **frekuensi tick per bin harga**, bukan volume transaksi riil. Harga mengikuti basis chart: Bid untuk chart Bid, Last untuk chart Last. Pada forex OTC, ini merupakan pendekatan POC berdasarkan aktivitas quote broker, bukan volume seluruh pasar forex.

Mode `NX_REAL_VOLUME` hanya menghitung tick bertanda transaksi buy/sell, memakai Last dan volume transaksi. Quote tick yang membawa ulang volume transaksi terakhir tidak dihitung ulang. Tidak ada fallback otomatis ke tick volume jika data transaksi tidak tersedia.

Histori diambil dengan `CopyTicksRange` untuk rentang selesai yang tepat. Jumlah tick harus cocok dengan jumlah `tick_volume` bar sumber. Pada real-volume mode, total volume transaksi juga harus cocok dengan real volume bar dalam toleransi numerik. Error, histori parsial, terlalu banyak tick, ketidakcocokan basis harga, atau profil tanpa satu puncak jelas menyebabkan entry dilewati.

Puncak harus minimal 1,2 kali rata-rata bin; dua bin dengan puncak setara ditolak. Profil memerlukan rentang minimal sebanyak jumlah bin dikali tick size. Batas default 500.000 tick per profil; naikkan hanya jika sumber daya dan histori memadai.

Buy membutuhkan harga di bawah POC, candle bullish dengan lower wick minimal 35% range, close lebih tinggi daripada candle sebelumnya, serta ADX <=25. Sell simetris. Range candle tetap dibatasi 0,4–2,5 ATR; filter body breakout 55% tidak digunakan untuk candle rejection POC.

Jarak close sinyal ke POC dan jarak entry aktual yang direncanakan ke POC masing-masing harus 0,5–2 ATR. TP menuju POC dan SL berjarak sama di sisi berlawanan, RR rencana 1:1. Tidak entry jika target sudah terlewati atau tidak memenuhi jarak stop broker.

Histori tick dapat memerlukan sinkronisasi terminal. EA tidak menebak histori yang hilang; entry dapat jarang terjadi atau tidak terjadi pada broker/data yang tidak sesuai.

## Perhitungan risiko

Default `InpRiskPercent=0.5` berarti 0,5% equity untuk **seluruh sinyal**, termasuk seluruh leg NXParsial. Batas input maksimal 5%; ini batas perangkat lunak, bukan rekomendasi penggunaan risiko 5%.

```
budget = equity × risk_percent / 100
unit_risk = abs(OrderCalcProfit(entry_adverse → SL, reference_lot)) / reference_lot
            + commission_round_trip_per_lot
entry_adverse = quote_entry + direction × deviation_points × point
lot_per_leg = floor_to_volume_step(budget / unit_risk / number_of_legs)
```

`reference_lot` menggunakan lot minimum simbol, sehingga tidak mengasumsikan 1 lot selalu valid. `OrderCalcProfit` mengembalikan nilai dalam mata uang akun. Tidak ada hardcode nilai pip, kontrak emas, atau kurs USD/JPY.

`InpCommissionPerLot` adalah estimasi komisi **round trip** dalam mata uang akun per 1 lot. Default 0 hanya benar untuk instrumen/akun tanpa komisi terpisah; isi sesuai broker jika ada biaya tersebut. Swap dan perubahan kurs setelah entry tidak dicadangkan.

Mode fixed lot memakai `InpFixedTotalLots` sebagai total volume sinyal, lalu dibagi rata dan dibulatkan turun. Batas risiko persentase **tetap berlaku**; EA menolak fixed lot yang melampauinya. Ini sengaja tidak mengizinkan fixed lot menonaktifkan batas risiko.

Contoh aritmetika ilustratif: equity USD10.000, risiko 0,5%, entry EURUSD 1,10020, SL 1,09720, contract size 100.000, deviation 10 points pada point 0,00001, dan komisi USD7/lot. Cadangan risiko adalah USD317/lot. Dengan volume step 0,01, single leg memakai 0,15 lot; NXParsial memakai 3 × 0,05 lot. Total estimasi risiko keduanya USD47,55. Implementasi tetap memakai hasil API broker, bukan angka ilustrasi ini.

SL/TP dikirim berdasarkan harga rencana sebelum fill. Harga aktual dapat berbeda sehingga **RR aktual tidak selalu tepat sama dengan RR rencana**. Risiko aktual dari fill diperiksa lagi; pelanggaran budget memicu rollback. EA tidak memodifikasi stop untuk menyamarkan slippage. Deviasi dapat diabaikan oleh beberapa mode eksekusi; gap SL juga dapat melampaui budget. Ini batas eksekusi pasar, bukan sesuatu yang bisa dijamin oleh rumus lot.

## Eksekusi, rollback, dan safety halt

EA membedakan `OrderCheck`, penerimaan permintaan, fill, dan posisi yang benar-benar terlihat. Nilai boolean `OrderSend` saja tidak dianggap bukti transaksi selesai. EA memeriksa retcode, deal, volume, harga, identifier posisi, dan SL/TP yang tercatat.

FOK diutamakan. Untuk strategi satu leg, IOC boleh digunakan bila broker mendukungnya, tetapi partial fill akan dibatalkan melalui rollback posisi yang sudah terbuka, bukan diterima sebagai transaksi normal. Return filling yang berpotensi menyisakan order tidak digunakan untuk entry.

Ada tiga status global per akun/server/simbol:

- `halt=0`: boleh mengevaluasi entry.
- `halt=1`: entry baru dihentikan. Digunakan saat permintaan sedang dikirim dan dipertahankan bila hasilnya ambigu, data fill belum dapat direkonsiliasi, atau rollback sudah selesai.
- `halt=2`: EA pemilik mencoba membatalkan order miliknya dan menutup posisi miliknya, lalu tetap berhenti untuk pemeriksaan. Percobaan dibatasi sekali per 5 detik saat ada tick.

Penolakan definitif pada leg pertama tidak mengirim ulang; EA melewati candle tersebut. Penolakan definitif setelah leg sebelumnya terisi memicu rollback. Timeout/status tidak pasti tidak diulang otomatis dan tidak dianggap tidak terjadi. Jika timeout muncul setelah satu leg NXParsial terisi, leg yang diketahui tetap memiliki SL/TP, tetapi basket tidak dianggap selesai; operator harus merekonsiliasi seluruh order/deal sebelum melanjutkan.

Rollback bukan transaksi atomik: penutupan bisa ditolak, mengalami slippage, menunggu quote atau trading aktif, dan tetap merugi. EA tidak menutup posisi manual atau magic lain. Pemilik rollback dicatat agar EA lain pada simbol tersebut tidak menganggap pekerjaan cleanup sudah selesai.

Cleanup hanya mencoba lagi setelah penolakan definitif atau partial close yang diketahui. Jika pembatalan/penutupan sendiri menghasilkan status ambigu, cleanup berhenti pada halt 1 untuk rekonsiliasi, tanpa mengulang permintaan yang mungkin masih diproses.

Identitas akun, server, simbol dan magic membentuk namespace state. Prefix dicetak di Journal. File lock mencegah dua instance dengan simbol/magic sama serta mengurutkan eksekusi strategi pada simbol yang sama **di terminal yang sama**. Tidak ada koordinasi dengan terminal/VPS lain. Gunakan satu pengelola trading per simbol; jangan menjalankan instance akun sama di terminal lain atau mengubah posisi EA secara bersamaan.

Global Variables terminal bukan arsip permanen: MT5 dapat menghapus variabel yang lama tidak diakses. Jangan mengandalkannya sebagai penyimpanan lintas migrasi terminal. Jangan mengganti magic, strategi, atau timeframe ketika masih ada posisi, halt, atau recovery yang perlu direkonsiliasi.

### Pemulihan halt secara manual

1. Matikan Algo Trading dan lepas EA yang relevan.
2. Cocokkan seluruh posisi, pending order, order history, dan deal history simbol tersebut, termasuk permintaan yang pernah timeout. Pastikan tidak ada permintaan yang masih diproses broker.
3. Bila rollback diperlukan, selesaikan penutupan/pembatalan atau pasang kembali **EA pemilik dengan magic yang sama** agar cleanup bisa mencoba lagi. SL/TP server tetap berlaku selama broker mempertahankannya.
4. Setelah simbol benar-benar bersih, buka Global Variables terminal. Gunakan nama `symbol safety key` yang tercetak di Journal. Ubah hanya variabel `*.halt` yang tepat menjadi 0. Jangan menghapus semua Global Variables.
5. Pertahankan state recovery dan penanda candle. Pasang kembali EA dengan input yang sama. EA menunggu candle berikutnya.

Reset bukan bukti bahwa order timeout tidak terjadi. Jangan reset sebelum hasil server sudah jelas. Paket tidak menyediakan tombol otomatis untuk menghapus status yang belum diperiksa.

## Batas penggunaan dan verifikasi

NXParsial memerlukan hedging. Strategi lain dapat berjalan pada hedging atau netting jika simbol tidak memiliki exposure lain. Simbol yang menghapus stop setiap rollover (`SYMBOL_ORDERS_DAILY`) ditolak. Stop standar tetap dapat mengalami gap, slippage, atau penolakan server.

Tidak ada batas drawdown portfolio, batas kerugian harian, filter berita, maupun korelasi lintas simbol. Risiko 0,5% adalah per sinyal, bukan batas agregat akun. Empat simbol yang berjalan bersama dapat menggunakan sekitar empat kali risiko per sinyal.

Filter close candle dan mutu data mencegah beberapa kesalahan logika, tetapi tidak menghilangkan false breakout atau menjamin arah harga. EMA, SNR rolling, ADX, dan tick-frequency POC adalah definisi operasional yang dipilih dari rancangan yang disetujui; bukan klaim bahwa strategi bermerek tertentu telah direplikasi dari kode aslinya.

Default bukan hasil optimasi. Belum ada bukti win rate, profit factor, kestabilan hasil, ataupun batas drawdown. Kompatibilitas sintaks, perilaku terminal, dan perilaku broker masih memerlukan verifikasi MT5 di tahap berikutnya yang Anda izinkan. Sesi ini berhenti pada sumber mentah dan pemeriksaan statis.

## Referensi API resmi

- [OrderCalcProfit](https://www.mql5.com/en/docs/trading/ordercalcprofit): estimasi hasil dalam mata uang akun.
- [OrderCheck](https://www.mql5.com/en/docs/trading/ordercheck): pemeriksaan permintaan sebelum pengiriman.
- [Trade return codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes): pemisahan fill, penolakan, partial, dan status ambigu.
- [Symbol properties](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants): point, tick size, lot step, stop level, filling, dan mode stop rollover.
- [CopyTicksRange](https://www.mql5.com/en/docs/series/copyticksrange) dan [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick): batas waktu histori, flag, dan data tick.
- [PositionClose](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionclose): hasil close harus diperiksa bersama retcode.
