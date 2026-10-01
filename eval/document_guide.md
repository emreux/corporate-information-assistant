# CIA Test Doküman Seti: Rehber ve Cevap Anahtarı

> ⚠️ **Bu dosyayı sisteme YÜKLEME / İNDEKSLEME.** İçinde bütün cevaplar var. Sisteme girerse her soruya bu dosyadan cevap bulur ve ölçümlerimiz anlamsızlaşır. Yeri: `evaluation/document_guide.md`. Dokümanların yeri: `data/samples/`.

Bütün şirket, kişi, müşteri, telefon ve e-posta bilgileri **kurgusaldır**. E-posta ve web adresleri bilerek `.example` uzantılıdır.

---

## 1. Şirket kartı

| Bilgi | Değer |
|---|---|
| Şirket | Kalkan Siber Güvenlik A.Ş. |
| Kuruluş | 2014, İstanbul |
| Genel merkez | Maslak / İstanbul (2025 Q1'e kadar Kozyatağı) |
| Bölge ofisleri | Ankara (Çankaya), İzmir (Bayraklı) |
| Çalışan / müşteri | 185 çalışan, 140+ aktif müşteri |
| Çalışan portalı | Kalkan Portal |
| VPN istemcisi | Kalkan Connect |

**Kişiler**

| İsim | Rol | Dahili |
|---|---|---|
| Kerem Arslan | Genel Müdür (CEO), Yönetim Kurulu Başkanı | – |
| Selin Aydemir | CISO | 4001 |
| Ayşe Demirci | İK Direktörü | – |
| Murat Kaya | SOC Müdürü | 4401 |
| **Murat Kayaalp** | Satış, Kıdemli Müşteri Temsilcisi (**Murat Kaya ile karıştırılmamalı**) | – |
| Onur Yalçın | BT Müdürü | 2001 |
| Av. Deniz Öztürk | Hukuk Birimi | 3050 |
| Elif Şahin | Kişisel Veri Koruma İrtibat Kişisi | 3055 |
| Zeynep Arıkan, Can Özdemir | Müşteri temsilcileri | – |
| Hakan Tunç | Yönetim Kurulu üyesi | – |

---

## 2. Dosyalar

| # | Dosya | Format | Önerilen `access_level` | İçerik |
|---|---|---|---|---|
| 1 | `IK_Politikasi_2025.pdf` | PDF, 2 sayfa | genel | **ESKİ** İK politikası (Rev.2, 01.01.2025) |
| 2 | `IK_Politikasi_2026.pdf` | PDF, 4 sayfa | genel | **GÜNCEL** İK politikası (Rev.3, 01.01.2026) |
| 3 | `Yan_Haklar_ve_Uzaktan_Calisma_Yonetmeligi.docx` | Word | genel | Yemek, ulaşım, sağlık, eğitim, uzaktan çalışma |
| 4 | `BT_Destek_Prosedurleri.docx` | Word | genel | Destek kanalları, VPN hata kodları, parola, cihaz |
| 5 | `SOC_Olay_Mudahale_Playbooklari.pdf` | PDF, 5 sayfa | genel | Önem seviyeleri, eskalasyon, PB-01…PB-05 |
| 6 | `Hizmet_Paketleri_ve_Fiyatlar.xlsx` | Excel, 2 sayfa (sheet) | genel | Paket kodları, fiyatlar, indirimler |
| 7 | `Musteri_Listesi.csv` | CSV (`;` ayraçlı) | genel | 27 satır, 20 müşteri |
| 8 | `Yonetici_Maas_Skalasi_2026.xlsx` | Excel | **yonetim_gizli** | Kademe bazlı maaş aralıkları |
| 9 | `Kalkan_Sirket_Tanitimi_2026.pptx` | PowerPoint, 7 slayt | genel | Şirket tanıtımı + konuşmacı notları |
| 10 | `Sik_Sorulan_Sorular.html` | HTML | genel | Çalışan SSS sayfası |
| 11 | `Yonetim_Kurulu_Tutanagi_2024-09_TARANMIS.pdf` | Taranmış PDF (metin katmanı yok) | genel | 12.09.2024 yönetim kurulu kararları |

---

## 3. Tuzaklar: Hangi dosya neyi test ediyor?

Parantez içindeki numaralar yol haritasındaki adımlardır.

| Kod | Tuzak | Nerede | Beklenen sorun | Çözecek adım |
|---|---|---|---|---|
| T1 | **Eski ve yeni versiyon** | #1 vs #2 (ayrıca #11 vs güncel durum) | Eski değerler cevaba karışır | Metadata (8), güncellik filtresi |
| T2 | **Olumsuzluk / istisna** | #2, #3, #5, #10 | "Çalışabilir" ile "çalışamaz" aynı vektöre düşer | Reranking (16) |
| T3 | **Birbirine çok benzeyen kodlar** | #4, #6, #5 | 203 ile 204, IT-07 ile IT-09 karışır | Hybrid/BM25 (15) |
| T4 | **Benzer isimler** | #5 (Murat Kaya / Murat Kayaalp) | Yanlış kişi bilgisi | Hybrid (15), reranking (16) |
| T5 | **Şirket jargonu** | PYS, EPS, Kalkan Connect, L1/L2/L3, P1–P4 | Embedding modeli bu kısaltmaları bilmez | BM25 (15), terim sözlüğü (17) |
| T6 | **Tablolar** | #2, #3, #4, #5, #6, #8 | Tablo satırları başlıklarından kopar | Doküman okuma (5), tablo işleme (6) |
| T7 | **Dağınık Excel** | #8: başlık 4. satırda, birleştirilmiş başlık hücreleri | Basit okuyucu "Unnamed: 1", "NaN" üretir | Tablo işleme (6) |
| T8 | **Not satırları tablonun içinde** | #6: "Fiyatlara KDV dahil değildir" | Not, bir paket satırı gibi okunur | Tablo işleme (6) |
| T9 | **Türkçe CSV** | #7: `;` ayraç, UTF-8 BOM, `gg.aa.yyyy` tarih | Virgülle okumaya çalışınca tek sütun çıkar | Tablo işleme (6) |
| T10 | **Tekrar eden başlık/altlık** | Tüm PDF'lerin her sayfası, Word başlık/altlıkları | "Sayfa 2", "Gizlilik Sınıfı" her chunk'a girer | Doküman okuma (5) |
| T11 | **Web sayfası kalıntıları** | #10: menü, çerez uyarısı, telif satırı | Menü metni chunk'lara karışır | Doküman okuma (5) |
| T12 | **Sayfa sonunda yalnız kalan başlık** | #2: "5.1 İzin Süreleri" 1. sayfanın sonunda, tablosu 2. sayfada | Başlık ile içerik ayrı chunk'lara düşer | Chunking + bağlam başlığı (7) |
| T13 | **Sayfalara bölünen tablo** | #5: önem seviyeleri tablosu 1. ve 2. sayfaya bölünmüş | P4 satırı tablodan kopuk kalabilir | Doküman okuma (5), chunking (7) |
| T14 | **Taranmış belge** | #11 | Metin katmanı yok, OCR gerekir. OCR dili Türkçe değilse "Güvenlik" → "Givenlik" | Doküman okuma (5) |
| T15 | **Konuşmacı notları** | #9, slayt 2 | Slaytta olmayan bilgi notlarda (üstelik "dışarıda paylaşılmasın" diyor) | Doküman okuma (5), metadata (8) |
| T16 | **Yetki** | #8 | Stajyer maaş bilgisine ulaşır | Yetki (14) |
| T17 | **Birden fazla dokümana yayılan cevap** | Deneme süresi (#2, #3, #10), maaş günü (#2, #10), yemek kartı (#3, #10) | Tekrar eden chunk'lar veya eksik cevap | Multi-query (18), atıf (13, 19) |
| T18 | **Hesaplama gerektiren soru** | "8 yıllık bir müdürün izni?" (#2: 22 + 2 = 24) | İki bölümü birleştirmek gerekir | Chunking (7), multi-query (18) |
| T19 | **Tablo üzerinde sayma/toplama** | #7 | "Kaç aktif müşteri var?" vektör RAG'in işi değil | **Bölüm 2** (agent + tool) |
| T20 | **Yakın ama cevapsız sorular** | "İzmir ofisinde otopark var mı?" (Maslak ve Ankara var, İzmir yok) | Model Ankara bilgisini İzmir'e uyarlar (uydurma) | "Cevap yok" kararı (16) |
| T21 | **Aynı sayı, farklı anlam** | `180.000 TL`: hem Müdür maaş alt sınırı (#8) hem Ankara ofis kira limiti (#11) | Yetki testinde yanlış alarm, BM25'te yanlış eşleşme | Evaluation (12) |
| T22 | **Excel'de sayı olarak saklanan tutarlar** | #6, #8 | Hücrede `85000` sayısı var, ekranda `85.000 TL` görünüyor. Okuyucu `85000` veya `85000.0` üretir; "85.000" araması ve anahtar bilgi kontrolü eşleşmez | Tablo işleme (6) |

> **T21 ve T22 için not:** Adım 12'de yetki sızıntısı testine "yasak bilgi" olarak `180.000` yazarsak, stajyerin gayet meşru şekilde görebildiği tutanak chunk'ı yüzünden **yanlış alarm** alırız. Yasak bilgiler maaş dosyasına özgü olmalı: `250.000`, `360.000`, `K4 - Müdür` gibi. Ayrıca Excel'deki tutarlar sayı olarak saklandığı için (T22), Adım 6'da biçimlendirmeyi çözene kadar hem `250.000` hem `250000` biçimini kontrol etmeliyiz; yoksa gerçek bir sızıntıyı **kaçırırız**.

---

## 4. Cevap anahtarı: Dokümanlardaki kritik bilgiler

### 4.1 İK Politikası: 2026 (güncel) ve 2025 (eski) farkları

| Konu | 2026 (Rev.3, GÜNCEL) | 2025 (Rev.2, ESKİ) |
|---|---|---|
| Yıllık izin, 1–5 yıl (5 dahil) | **16 gün** | 14 gün |
| Yıllık izin, 5–15 yıl | **22 gün** | 20 gün |
| Yıllık izin, 15 yıl ve üzeri | **28 gün** | 26 gün |
| Yönetici (müdür ve üzeri) ilave izni | **+2 gün** | Yok |
| İzin talebi | **Kalkan Portal, 10 iş günü önce**; 5 iş gününden uzun izinde İK onayı da gerekir | KLK-IK-F03 formu, e-posta, 2 hafta önce |
| İzin devri | **En fazla 5 gün**, 31 Mart'a kadar kullanılmalı | Devredilemez |
| Mazeret izinleri | Evlilik 3, babalık 5, vefat 3, **taşınma 1 gün, doğum günü yarım gün** | Evlilik 3, babalık 5, vefat 3 |
| Esnek başlangıç | **08:00–10:00 arası**, çekirdek saat 10:00–16:00; SOC vardiyası hariç | Yok |
| Performans değerlendirme | **Yılda 2 kez (Haziran, Aralık)**, PYS; hedef girişi 31 Ocak'a kadar | Yılda 1 kez (Aralık) |
| Disiplin itiraz süresi | **7 iş günü** | 5 iş günü |
| Yemek kartı | Yan Haklar Yönetmeliği'ne taşındı: **3.250 TL** | 2.750 TL |
| Uzaktan çalışma | Yan Haklar Yönetmeliği'ne taşındı: **haftada 2 gün** | Haftada 1 gün |
| Sağlık sigortası kapsamı | Yönetmelik: **çalışan + eş + 25 yaş altı çocuklar** | Çalışan + eş |

Değişmeyenler: Haftalık 45 saat, mesai 09:00–18:00, öğle arası 12:30–13:30, deneme süresi 2 ay (deneme süresinde yıllık izin yok, mazeret izni var), fazla mesai %50 zamlı ve yıllık 270 saat, maaş her ayın 15'i.

Sadece 2026'da olanlar: Fazla mesai için önceden yönetici onayı şart, serbest zaman seçeneği (her saat için 1,5 saat, 6 ay içinde), SOC resmi tatil ek ödemesi, izinlerin bölünmesinde bir parça en az 10 gün, bordro "Kalkan Portal > Bordrolarım", maaş günü hafta sonuna denk gelirse bir önceki iş günü, ihbar süreleri (6 aydan az: 2 hafta, 6 ay–1,5 yıl: 4 hafta, 1,5–3 yıl: 6 hafta, 3 yıldan fazla: 8 hafta), oryantasyon ve ilk hafta zorunlu güvenlik eğitimi, "Takım liderleri yönetici tanımına dahil değildir".

### 4.2 Yan Haklar ve Uzaktan Çalışma Yönetmeliği (KLK-IK-YON-04)

- Yemek kartı **3.250 TL/ay**, her ayın ilk iş günü; anlaşmalı restoranlarda %10 indirim; Maslak'ta yemekhane yok.
- Servis yalnızca Maslak (Kadıköy, Bakırköy, Ataşehir; 07:45 kalkış, 18:10 dönüş). Ankara ve İzmir'de servis yok, **1.500 TL/ay yol yardımı**.
- Sağlık sigortası: çalışan, eş, **25 yaşını doldurmamış** çocuklar. Anne-baba **kapsam dışı**, ancak prim çalışan tarafından ödenerek eklenebilir. Check-up: 35 yaş ve üzeri.
- Eğitim bütçesi **7.500 TL/yıl**, devretmez. Sertifika sınav ücreti (OSCP, CISSP, CEH, GCIH, Security+) şirketten; şartlar: **yönetici onayı + 12 ay çalışma taahhüdü**. 2. deneme de şirketten, 3. ve sonrası çalışandan.
- Spor salonu **%50, en fazla 1.000 TL/ay**. Doğum yardımı **10.000 TL**.
- Uzaktan çalışma: **haftada en fazla 2 gün**; **Salı ve Perşembe ofis günü** (uzaktan çalışılamaz); **deneme süresindekiler uzaktan çalışamaz**; **SOC L1 analistleri uzaktan çalışamaz** (L2/L3 nöbet dışında genel kurala tabi).
- Ev ofis desteği **5.000 TL tek seferlik** + internet desteği **400 TL/ay**.
- Güvenlik: sadece şirket cihazı, Kalkan Connect VPN, halka açık Wi-Fi'da VPN'siz bağlanılmaz, ekran kilidi 5 dk, müşteri verisi evde yazdırılamaz.
- Yurt dışından uzaktan çalışma: **yılda en fazla 20 iş günü**, İK ve BT'nin önceden yazılı onayı; verisi Türkiye dışına çıkamayan projelerde yasak.

### 4.3 BT Destek Prosedürleri (KLK-BT-PRS-02)

| Hata Kodu | Anlamı | Çözüm |
|---|---|---|
| **VPN-ERR-203** | Sertifika süresi dolmuş | Kalkan Portal > Güvenlik > Sertifikalarım'dan yenile, Kalkan Connect'i tamamen kapatıp yeniden başlat |
| **VPN-ERR-204** | DNS yapılandırması hatalı | Kullanıcı çözemez; BT Destek'e "Ağ" kategorisinde talep, ekran görüntüsüyle |
| **VPN-ERR-230** | MFA zaman aşımı | Telefon saatini "otomatik" yap, Authenticator'da zaman senkronizasyonu |
| **VPN-ERR-301** | Eş zamanlı bağlantı lisans limiti | Diğer cihazdaki VPN oturumunu kapat |

- Destek: Kalkan Portal > BT Destek > Yeni Talep; bt-destek@kalkansiber.example; **dahili 2020** (hafta içi 08:00–20:00); acil güvenlik olayları **dahili 4444** (7/24 SOC).
- Öncelikler: Kritik 1 saat, Yüksek 4 saat, Normal 1 iş günü, Düşük 3 iş günü.
- Parola: en az **14 karakter**, **90 günde bir** değişim, **son 12** parola tekrar kullanılamaz, **5 hatalı denemede 30 dk kilit**, MFA zorunlu. Parolamı Unuttum: portal giriş ekranı + SMS; telefon değiştiyse dahili 2020, görüntülü görüşme.
- **KLK-IT-07** = Donanım Talep Formu, **KLK-IT-09** = Yazılım Lisans Talep Formu.
- Dizüstü: standart 14 inç; SOC analistleri ve geliştiriciler 16 inç, 32 GB.
- Kayıp/çalıntı cihaz: **en geç 1 saat içinde dahili 4444**; bildirimi ertelemeyin; çalınmada emniyet tutanağı.

### 4.4 SOC Olay Müdahale Playbook'ları (KLK-SOC-PB-00 Rev.5)

| Seviye | İlk Yanıt | Kontrol Altına Alma |
|---|---|---|
| P1 Kritik | **15 dakika** | 4 saat |
| P2 Yüksek | 1 saat | 8 saat |
| P3 Orta | 4 saat | 2 iş günü |
| P4 Düşük | 1 iş günü | 5 iş günü |

- Belirsizse **daha yüksek** seviye seçilir.
- Eskalasyon: P1 → SOC Müdürü ve CISO **derhal telefonla** (e-posta yetmez), müşteri **30 dk** içinde. P2 → SOC Müdürü 1 saat, müşteri 2 saat. P3/P4 → vardiya lideri.
- **PB-01 Fidye yazılımı (P1):** EDR ile izole et, **cihazı kapatma/yeniden başlatma** (bellek imajı), yedekleri koru; **SOC saldırganla hiçbir koşulda iletişime geçmez**, ödeme kararı müşteri yönetimi ve hukukta.
- **PB-02 Oltalama:** varsayılan P3, tıklama + kimlik bilgisi girişi varsa P2; bağlantıları kendi tarayıcında açma.
- **PB-03 Hesap ele geçirme:** varsayılan P2, **admin hesabıysa P1**; son 30 günlük oturum kayıtları.
- **PB-04 DDoS:** kesinti varsa P1, yavaşlama P2; dikkat dağıtma saldırısı kontrolü.
- **PB-05 Veri sızıntısı:** aktifse P1; SOC **48 saat** içinde Hukuk'a yazılı rapor; Kurul bildirimi **Hukuk Birimi** tarafından en geç **72 saat**; SOC Kurul'a doğrudan bildirim yapmaz.
- Kanıt: SHA-256, şifreli kanıt deposu, **2 yıl** saklama. Post-mortem: P1/P2 sonrası **5 iş günü** içinde.

### 4.5 Hizmet Paketleri (fiyatlar KDV hariç, 01.07.2026 itibarıyla)

| Kod | Paket | Ücret | P1 İlk Yanıt | Min. Süre |
|---|---|---|---|---|
| KLK-SOC-100 | Temel SOC İzleme (8/5, 500 EPS) | 45.000 TL/ay | 1 saat | 12 ay |
| KLK-SOC-110 | SOC İzleme Plus (7/24, 2.000 EPS) | 85.000 TL/ay | 15 dk | 12 ay |
| KLK-SOC-120 | SOC İzleme Kurumsal (10.000 EPS) | 160.000 TL/ay | 15 dk | 24 ay |
| KLK-EDR-200 | Yönetilen EDR (250 uç nokta) | 38.000 TL/ay | 30 dk | 12 ay |
| KLK-EDR-210 | Yönetilen EDR Plus (1.000 uç nokta) | 95.000 TL/ay | 15 dk | 12 ay |
| KLK-PT-300 | Sızma Testi - Web | 120.000 TL tek seferlik | – | – |
| KLK-PT-310 | Sızma Testi - İç Ağ | 180.000 TL tek seferlik | – | – |
| KLK-PT-320 | Sızma Testi - Mobil | 140.000 TL tek seferlik | – | – |
| KLK-EDU-400 | Farkındalık Eğitimi | 750 TL/kişi/yıl | – | 12 ay |
| KLK-IR-500 | Olay Müdahale Hazır Bekleme (40 saat) | 300.000 TL/yıl | 30 dk | 12 ay |

İndirimler: 24 ay %12, 36 ay %18, 3+ paket %5 ek (taahhüt indirimiyle birleşir), kamu %10 (**diğer indirimlerle birleşmez**).

### 4.6 Maaş Skalası (GİZLİ, yalnızca üst yönetim ve İK)

K1 Uzman 75.000–95.000 · K2 Kıdemli Uzman 95.000–130.000 · K3 Takım Lideri 130.000–170.000 · **K4 Müdür 180.000–250.000** · K5 Direktör 260.000–360.000 · K6 Üst Yönetim: Yönetim Kurulu kararıyla. Prim hedefleri %5/%8/%10/%15/%20/%25. Zam dönemi Ocak ve Temmuz.

### 4.7 Diğerleri

- **Tanıtım (pptx):** 2014, Maslak, Ankara ve İzmir ofisleri, 185 çalışan, 140+ müşteri, ISO 27001 ve ISO 9001, %97 yenileme oranı (2025), ortalama P1 ilk yanıt 11 dk, 210 sızma testi. **Konuşmacı notu (slayt 2):** 2026 sonu hedefi 200 müşteri ve 230 çalışan, henüz kamuya açıklanmadı.
- **SSS (html):** Maaş 15'inde; bordro son 24 ay; yemek kartı ilk iş günü; yarım gün izin mümkün (12:30 öncesi / 13:30 sonrası); **Maslak'ta 40 araçlık otopark** (İK sırasına göre); **Ankara'da otopark yok**, anlaşmalı otoparkta %30 indirim; misafir 1 gün önce portaldan, **SOC katına misafir alınmaz**; kartvizit 5 iş günü; şüpheli e-posta → "Oltalama Bildir" düğmesi.
- **Tutanak (taranmış, 12.09.2024):** Genel merkezin Kozyatağı'ndan Maslak'a taşınması (2025 Q1); yeni SIEM için **4.200.000 TL**; SOC'a 2025'te **6 yeni L1 analist**; Ankara ofisi Haziran 2025 Çankaya, kira üst limiti **180.000 TL/ay**; yan hakların gözden geçirilmesi.

---

## 5. Başlangıç soru seti (Adım 12'de genişleteceğiz)

| # | Tür | Rol | Soru | Beklenen cevap | Anahtar bilgi |
|---|---|---|---|---|---|
| 1 | olgusal | calisan | 8 yıllık bir çalışanın kaç gün yıllık izni var? | 22 gün (2025'teki 20 değil) | `22 gün` |
| 2 | güncellik | calisan | Yemek kartına ne kadar yükleniyor? | 3.250 TL/ay (2.750 değil) | `3.250` |
| 3 | hesaplama | calisan | 8 yıllık kıdemi olan bir müdürün yıllık izni kaç gün? | 24 gün (22 + 2) | `24 gün` |
| 4 | tuzak | calisan | 8 yıllık bir takım liderine ilave izin verilir mi? | Hayır, takım liderleri yönetici değil; 22 gün | `Takım liderleri` |
| 5 | olumsuzluk | calisan | Deneme süresindeyim, evden çalışabilir miyim? | Hayır | `deneme süresindeki çalışanlar uzaktan çalışamaz` |
| 6 | olumsuzluk | calisan | Deneme süresinde mazeret izni kullanabilir miyim? | Evet (yıllık izin kullanılamaz) | `Mazeret izinleri` |
| 7 | olumsuzluk | calisan | SOC L1 analistiyim, uzaktan çalışabilir miyim? | Hayır | `L1 analistler uzaktan çalışamaz` |
| 8 | farklı ifade | calisan | Evden çalışmak için masa sandalye almaya destek var mı? | 5.000 TL tek seferlik ev ofis desteği | `5.000` |
| 9 | kod | calisan | vpn err 203 aldım napcam | Sertifikayı portaldan yenile, istemciyi yeniden başlat | `Sertifikalarım` |
| 10 | kod | calisan | VPN-ERR-204 ne demek? | DNS hatası, BT'ye "Ağ" kategorisinde talep | `DNS` |
| 11 | kod | calisan | KLK-IT-09 formu ne için kullanılır? | Yazılım lisans talebi (IT-07 donanım) | `Yazılım Lisans` |
| 12 | kod | calisan | KLK-SOC-110 paketinin aylık ücreti ne kadar? | 85.000 TL (KDV hariç) | `85.000` (bkz. T22) |
| 13 | isim | calisan | Murat Kaya'nın dahili numarası kaç? | 4401 (Murat Kayaalp ile karıştırılmamalı) | `4401` |
| 14 | jargon | calisan | PYS'ye hedeflerimi ne zamana kadar girmeliyim? | 31 Ocak | `31 Ocak` |
| 15 | olumsuzluk | calisan | Fidye yazılımı bulaşan bilgisayarı kapatmalı mıyım? | Hayır, kapatılmamalı (bellek imajı) | `kapatmayın` |
| 16 | çok parçalı | calisan | Yemek kartı ne kadar ve maaşlar ayın kaçında yatıyor? | 3.250 TL; ayın 15'i | `3.250`, `15'inde` |
| 17 | çok parçalı | calisan | Parola en az kaç karakter olmalı ve kaç günde bir değişir? | 14 karakter, 90 gün | `14 karakter`, `90 gün` |
| 18 | tablo | calisan | Kamu kurumu indirimi taahhüt indirimiyle birleşir mi? | Hayır | `birleştirilemez` |
| 19 | OCR | calisan | Yeni SIEM geçişi için ne kadar bütçe onaylandı? | 4.200.000 TL | `4.200.000` |
| 20 | notlar | calisan | Şirketin 2026 sonu müşteri hedefi nedir? | 200 aktif müşteri (konuşmacı notunda) | `200 aktif müşteri` |
| 21 | takip | calisan | Geçmiş: "Çalışanların yıllık izni kaç gün?" → "peki yöneticiler için?" | Kıdeme göre süre + 2 gün | `ilave` |
| 22 | cevaplanamaz | calisan | İzmir ofisinde otopark var mı? | Dokümanlarda bilgi yok | – |
| 23 | cevaplanamaz | calisan | Şirketin borsa değeri ne kadar? | Dokümanlarda bilgi yok | – |
| 24 | yetki | stajyer | Müdürlerin maaş aralığı ne kadar? | Bilgi yok / erişemez | yasak: `250.000`, `250000`, `K4 - Müdür` |
| 25 | yetki | mudur | Müdürlerin maaş aralığı ne kadar? | 180.000 – 250.000 TL brüt | `250.000` |
| 26 | agent (Bölüm 2) | calisan | Kaç aktif müşterimiz var? | 11 firma (14 "Aktif" satır; bazı firmaların birden fazla paketi var) | Vektör RAG ile güvenilir cevaplanamaz |

> Soru 26'yı RAG sistemi büyük ihtimalle yanlış cevaplayacak. Bu beklenen bir başarısızlık: Sayma işi tablo üzerinde sorgu gerektirir. Üstelik "satır sayısı" (14) ile "firma sayısı" (11) farklı, yani doğru sayı için sorgunun da doğru yazılması gerekiyor. Bölüm 2'de bunu agent'a tablo sorgulama tool'u vererek çözeceğiz.
