# Alur dan tinjauan sumber

## Batas bukti

Dokumen ini mencatat spesifikasi dan hasil tinjauan statis sumber. Tidak ada compiler, terminal MT5, Strategy Tester, emulator MQL5, atau koneksi broker yang dijalankan. Skenario di bawah adalah penelusuran cabang kode dan kriteria penerimaan pengujian MT5 berikutnya, **bukan hasil backtest atau hasil eksekusi skenario di MT5**.

## Pemetaan modul

| Modul | Tanggung jawab |
|---|---|
| `NXInputs.mqh` | Input umum dan input khusus strategi melalui macro |
| `NXSignals.mqh` | Data candle selesai, indikator, breakout, SNR, dan profil tick |
| `NXTrade.mqh` | State, lock, quote, rencana harga, lot, margin, pengiriman, dan rollback |
| `NXEngine.mqh` | Validasi input, event EA, satu evaluasi per bar, dan recovery |

File standalone menggabungkan teks modul yang sama. Penggabungan teks tidak membuat executable atau membuktikan validitas sintaks MQL5.

## Alur eksekusi

```text
OnInit
  -> validasi input, properti simbol, jenis akun
  -> tolak simbol yang menghapus protective stops saat rollover
  -> buka exclusive instance lock
  -> siapkan state persisten dan handle indikator
  -> catat candle berjalan; tidak entry pada pemasangan

OnTick
  -> ambil exclusive symbol lock (terminal lokal)
  -> jika halt=2 dan EA adalah pemilik: coba cleanup
  -> jika candle baru: lanjut satu kali
     -> halt / exposure lain / trading tidak aktif / terlambat? lewati
     -> simpan penanda evaluasi candle sebelum menilai sinyal
     -> NXReverse: rekonsiliasi histori posisi dan status recovery
     -> muat candle selesai dan indikator
     -> bangun sinyal dan terapkan filter strategi
     -> NXReverse: validasi arah dan waktu konfirmasi recovery
     -> periksa quote, drift entry, level breakout masih bertahan
     -> rencanakan SL/TP di tick grid
     -> hitung total risiko, lot per leg, dan kebutuhan margin
     -> preflight seluruh leg dengan OrderCheck
     -> simpan pemilik dan halt=1 sebelum pengiriman
     -> untuk setiap leg:
        -> cocokkan exposure dengan leg sebelumnya yang sudah dibuka
        -> periksa ulang quote, waktu entry, stop, risiko, OrderCheck
        -> OrderSend sekali; tidak ada automatic resend
        -> validasi retcode + deal + volume + risiko fill + protective stops
     -> hanya bila semua leg terkonfirmasi: halt=0
  -> lepaskan symbol lock

OnDeinit
  -> lepas handle indikator dan file instance lock
  -> flush state; tidak menutup posisi normal
```

Kegagalan memuat data pada tick pertama candle menyebabkan candle dilewati; EA tidak berulang kali menebak sinyal sambil menunggu indikator. Pengecekan waktu entry diulang setelah pembacaan histori yang mungkin lambat.

## Definisi R

R harga adalah `abs(entry_rencana - SL)`. Target dihitung `entry_rencana + side × multiplier × R`, kemudian dibulatkan ke tick size. R uang basket adalah total estimasi kerugian semua leg pada SL, dengan cadangan deviasi entry serta estimasi komisi. Keduanya tidak disamakan setelah biaya/slippage.

NXBreak dan NXParsial menggunakan SL fixed points yang dibulatkan ke tick grid. NXReverse menggunakan swing stop dan tidak mengubahnya untuk memaksa entry. NX-POC menggunakan target profil yang sudah selesai; jarak SL dicerminkan dari jarak target.

## Skenario yang ditelusuri pada sumber

| Skenario | Cabang/perilaku yang dituju |
|---|---|
| High candle berjalan menembus level lalu kembali | Tidak entry intrabar; candle 0 tidak membentuk sinyal |
| Wick menembus, close tidak melewati buffer | `NXBreakDirection` menghasilkan 0 |
| Gap open sudah di luar level | Persyaratan crossing open tidak terpenuhi |
| Body kecil atau range ekstrem | Filter kualitas menolak breakout |
| Quote sudah lama, spread tinggi, entry terlalu jauh | `NXQuote` / batas drift menolak |
| Data profil selesai dimuat setelah window entry | `NXEntryWindow` menolak sebelum pengiriman |
| SL terlalu dekat dengan Bid/Ask saat ini | `NXStopsValid` menolak; stop tidak dilebarkan diam-diam |
| Lot hasil risiko lebih kecil dari minimum | Entry dilewati; lot tidak dibulatkan naik |
| Fixed total lot melampaui risk budget | Entry ditolak |
| NXParsial: total risiko 0,5% | Volume per leg dihitung dari budget total dibagi tiga |
| NXParsial: akun netting | `OnInit` menolak |
| NXParsial: hanya IOC | `NXFillMode` menolak sebelum entry |
| `OrderCheck=true`, retcode 0 | Preflight diterima; tidak dianggap fill |
| `OrderCheck=false` atau retcode penolakan | Tidak mengirim leg tersebut |
| Leg pertama ditolak definitif | Tidak retry pada candle sama; halt dapat kembali 0 |
| Leg kedua ditolak definitif setelah leg pertama terisi | Halt 2, rollback posisi milik EA, lalu tetap halt |
| Leg sebelumnya keluar sebelum leg berikutnya dikirim | `NXBatchExposureMatches` gagal; batch dihentikan |
| `OrderSend=true`, retcode PLACED atau deal 0 | Status ambigu; entry berikutnya diblokir |
| Timeout atau error tidak jelas | Tidak resend; halt 1 dipertahankan untuk rekonsiliasi |
| Partial fill IOC | Halt 2; volume parsial tidak dianggap transaksi normal |
| Harga fill membuat risiko melampaui budget | Halt 2; coba tutup posisi |
| Deal tercatat tetapi posisi belum terlihat | Tidak mengasumsikan fill selesai dikelola; halt dipertahankan |
| Posisi terlihat tanpa SL/TP sesuai permintaan | Halt 2; coba tutup posisi |
| Restart setelah journal sebelum hasil order diketahui | Halt tersimpan; tidak membuka siklus baru |
| Dua chart simbol/magic sama | Exclusive instance file menolak instance kedua |
| Dua EA berbeda pada simbol sama | Symbol lock mengurutkan pengiriman; exposure gate memblokir siklus kedua |
| EA lain melihat halt rollback milik NXParsial | Tidak menjalankan/menyelesaikan cleanup atas nama EA pemilik |
| SL normal NXReverse | Histori posisi harus flat dan settled, lalu recovery berlawanan diaktifkan |
| TP/close manual/stop-out NXReverse | Tidak mengaktifkan recovery |
| Recovery NXReverse SL | Mode recovery menghalangi recovery berantai |
| Sinyal berlawanan terbentuk sebelum SL | Tidak sah sebagai konfirmasi recovery |
| Pending recovery lewat batas bar | Kembali ke strategi normal |
| POC: histori tick tidak cocok dengan bar | Tidak entry; tidak membangun distribusi volume rekaan |
| POC: quote membawa volume transaksi lama | Tidak menghitung ulang volume pada real-volume mode |
| POC: real volume kosong | Tidak mengganti sumber menjadi tick volume |
| POC: dua puncak terbesar sama / distribusi terlalu rata | Tidak entry |

## Peninjauan API

- Preflight dan fill memakai kriteria berbeda. [Contoh resmi OrderCheck](https://www.mql5.com/en/docs/trading/ordercheck) menampilkan retcode preflight 0, sedangkan pengiriman yang selesai memakai 10009. Implementasi menerima 0 atau DONE hanya pada preflight; fill tetap memerlukan DONE dan deal nonzero.
- Protective stop buy diperiksa terhadap Bid, sell terhadap Ask. Sizing memakai harga entry Ask/Bid, sehingga spread tidak hilang dari estimasi kerugian.
- Pembulatan lot selalu turun; harga mengikuti `SYMBOL_TRADE_TICK_SIZE`, bukan sekadar jumlah digit.
- Perhitungan mata uang memakai `OrderCalcProfit`, bukan perkalian pip value statis yang dapat salah pada cross-currency/CFD.
- Histori tick dibatasi sebelum candle pemicu mulai, menggunakan interval milidetik inklusif API.
- Real-volume profile hanya menimbang tick transaksi, agar volume yang tersalin pada quote tick tidak dihitung dua kali.
- Ticket histori disimpan sebagai dua bagian 32 bit untuk menghindari kehilangan presisi ulong dalam Global Variables bertipe double.

## Hal yang belum terverifikasi

Pemeriksaan statis tidak membuktikan bahwa kode diterima oleh build MetaEditor tertentu. Belum ada bukti atas urutan event trade, latensi visibility posisi, konsistensi tick/bar broker tertentu, komisi aktual, filling policy broker, atau hasil cleanup di pasar nyata.

Jika nanti tahap pengujian diizinkan, kriteria utamanya: keempat EA diterima MetaEditor; kasus penolakan tidak membuat order duplikat; risk basket sesuai toleransi mata uang akun; partial/timeout tidak meneruskan basket; recovery tidak berantai; dan POC menolak data tak lengkap. Setelah itu baru uji historis real ticks, periode out-of-sample, serta forward demo. Tidak satu pun tahap tersebut dilakukan dalam penyerahan sumber ini.
