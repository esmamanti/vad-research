# Deney ve tablo incelemesi

Bu inceleme EXP-01–07'nin kaynak kodunu, notebook çıktılarını ve CSV dosyalarını kapsar. EXP-01–06 yerel önbellekteki LibriSpeech VAD verisiyle yeniden çalıştırılır. EXP-07 tabloları mevcut CSV'lerden yeniden düzenlenir; bu aşamada EXP-07 modelleri yeniden eğitilmez. Sayısal kontrollerin envanteri `audit_inventory.csv`, yeniden üretim karşılaştırması `audit_reproduction.csv` dosyalarındadır.

Doğrulama tamamlandı: EXP-01–06 yeniden çalıştı; aktif eski sonuçların sayıları yeniden üretildi. Git karşılaştırması `audit_git_baseline.csv` dosyasında: 17 mevcut dosya sayısal olarak aynı; bu sayıya dokunulmayan iki eski dosya da dahildir, onların yeniden üretilmiş olduğu anlamına gelmez. EXP-05 hop işlem sıklığı ayrı dosyaya taşındı, accuracy değerleri değişmedi. 43 CSV dosyasının envanteri çıkarıldı; karışıklık sayısı bulunan tablolarda türetilebilir metrikler kontrol edildi. Yedi notebook'un Python hücreleri sözdizimi kontrolünden geçti; kaydedilmiş hata çıktısı yok. Dört tablo gösterimleri ve EXP-06'nın 4 × 4 grafiği doğrulandı. Windows paralel havuz erişim kısıtı nedeniyle EXP-06 RF yeniden üretimi aynı seed/modelle tek iş parçacığında yapıldı; eski sayılar aynen elde edildi.

## Dört yerine iki gösterim sorunu

İki ayrı eksiklik saptandı:

- **EXP-06 özellik dağılımları:** Dört ortam (`clean`, `white 10 dB`, `fluctuating 10 dB`, `babble 10 dB`) hazırlanıyor; grafik kodu yalnız ilk iki ortamı çiziyordu. Dört ortam × dört özellik olarak düzeltildi. Ayrıca tek özellik AUC'leri her ortam için ayrı dört tablo halinde gösteriliyor. Bunlar ham özellik değerleri değil; ayırma gücü tabloları. Ham değerler için medyan/p10/p90 özeti eklendi.
- **EXP-07 özellik karşılaştırması:** İki model × iki metrik birleşik, çok başlıklı bir tablodaydı. Ayrı LR-AUC, LR-miss, RF-AUC ve RF-miss tabloları oluşturuldu. Yeni CSV'ler `exp07_feature_{lr|rf}_{auc|miss}_dev.csv` olarak kaydedildi.

EXP-07 test özeti zaten kaynak kodda dört çıktı üretiyordu: balanced accuracy, miss, FAR ve salt gürültü FAR. Sonuncusu Series biçimindeydi; tablo biçimine dönüştürüldü. Bu dört özetin ayrı CSV'leri de eklendi. Bunlar dört farklı veri seti değil, aynı testin dört görünümüdür.

## Deney bazında değerlendirme

| Deney | Gerçekte ölçtüğü soru | Sonuç ve yorum sınırı |
| --- | --- | --- |
| EXP-01 | Tek temiz kayıtta RMS eşiği nasıl etkiliyor? | 0,01 eşikte miss %17,1; 0,05'te %90,2. Eşik yükselince zayıf konuşma kayboluyor. En iyi eşik aynı kayıt etiketleriyle seçiliyor; genel test başarısı değil. |
| EXP-02 | Tek kayıtta farklı gürültü ve SNR'lerde enerji eşiği | Beyaz gürültü 10 dB'de temizden seçilmiş sabit eşik FAR %100'e çıkıyor. Konuşmayı kaçırmaması başarı değil: her şeyi konuşma kabul ediyor. 6 dB adaptif marjda miss yaklaşık %38,6, FAR %0. |
| EXP-03 | Tek kayıtta adaptif marj yöntemleri | Beyaz 10 dB'de spread-based balanced accuracy yaklaşık %92,9; dalgalanan gürültüde yaklaşık %81,4. Tek koşuldaki kazanç bütün ortamlar için geçerli değil. Oracle gerçek etiketleri görerek eşik seçiyor. |
| EXP-04 | Farklı konuşmacılara eşik yöntemlerinin aktarımı | Dev'de seçilen k=1,5 ile koşullar ortalaması balanced accuracy yaklaşık %82,94; k=2 ile %81,59. Bu ortalama koşulların eşit ağırlıklı ortalaması. Konuşmacı ayrımı var. Sonradan eklenen robust karşılaştırması artık ayrı CSV'de. |
| EXP-05 | Frame/hop'un enerji tabanlı karara etkisi | Label-free temiz koşulda denenmiş boylar arasında 64 ms yaklaşık %91,29; dalgalanan gürültüde 128 ms yaklaşık %83,24. Bu sonuç ML modeli için optimum frame kanıtı değil. |
| EXP-06 | Özellik ekleme/çıkarma, LR/RF karşılaştırması | RF tüm özelliklerle koşul ortalaması oracle miss %13,72; LR %19,52. Her test ortamında eşik test negatiflerinden seçiliyor. Bu yüzden sabit eşikli üretim başarısı olarak sunulamaz. |
| EXP-07 | İstenen veri setinden temiz/gürültü karışımlarıyla ML VAD | Eğitim çeşitliliği bazı test koşullarında yararlı; mutlak enerji düşük kayıt seviyesinde büyük sorun yaratıyor. Frame referansları temiz kayıttan enerjiyle oluşturulmuş otomatik etiketler. |

### EXP-05'te başlık veriden daha iddialıydı

“Uzun frame sınırları bozar” her koşul ve her boy için doğru değil. Temiz ses oracle sınır hatası 8 ms'de %18,87, 64 ms'de %14,86, 128 ms'de %16,06, 256 ms'de %23,65. İlk büyütmeler iyileştiriyor, sonra bozulma başlıyor. Dalgalanan gürültüde ise sınır hatası 8–256 ms aralığında azalıyor. Başlık bu farklılığı anlatmalı.

Hop değerlendirmesi 32 ms aralıklı referans bloklarına en yakın frame'i eşleştiriyor. 4 ms ile 8 ms hop'un yakın sonuç vermesi, 4 ms'nin konuşma başlangıcında hiç yararı olmadığı anlamına gelmez; referans zaman çözünürlüğü daha kaba. `frames per second` satırı işlem sıklığıdır, accuracy değildir; ayrı tutulmalıdır.

### EXP-06'da özelliklerin etkisi ortama bağlı

LR'ye MFCC eklendiğinde temiz koşul oracle miss %6,56 → %5,44, dalgalanan gürültü %30,81 → %27,51. Beyaz gürültüde ise %9,36 → %12,07 ile kötüleşiyor. Dolayısıyla “MFCC her koşulda daha iyi” sonucu çıkmaz.

Tüm özelliklerden enerjiyi çıkarınca LR ortalama oracle miss %19,52 → %40,48. Buna rağmen dalgalanan gürültüde %27,51 → %24,63 iyileşiyor. Bir özellik bazı ortamları iyileştirip bazılarını bozabilir. Önemi tek ortalamayla açıklamamak gerekiyor.

### EXP-07'de dev tablosu ve final model aynı ayar değil

Özellik ablation'ı context=0, smoothing=1 ile yapılıyor. Nihai modeller context=5 ve smoothing=21 kullanıyor. AUC/miss ablation tablosunu doğrudan nihai modelin test sonucu gibi okumamak gerekir.

Context/smoothing tablosunda LR için k=10/window=31 oracle dev miss yaklaşık %16,2; final ayar k=5/window=21 ise yaklaşık %19,5. Kodun neden daha kısa ayarı seçtiğine ilişkin yazılı hedef veya gecikme bütçesi yok. Daha kısa ayar bir tercih olabilir ama “en yüksek başarı veren ayar” diye sunulmamalı. Context ve merkezli smoothing geleceği kullanır.

Temiz eğitimli LR'nin dev kalibrasyon eşiği ekranda 1,000'a yuvarlanıyor. Bu, modeli bütün gürültülü dev negatiflerinde düşük FAR'a zorlayan çok katı bir eşik. Temiz testte yüksek miss'in bir kısmı bu çalışma noktasının etkisi olabilir; yalnız modelin özellik öğrenmesine bağlanmamalı. Her modelin eşik ve kalibrasyon dağılımı raporda bulunmalı.

## Düzeltilen çıktı/kayıt sorunları

1. EXP-04 final robust tablosu kaydedilmiyordu: `exp04_robust_methods_by_condition.csv` eklendi. Önceki tablo tarihsel karşılaştırma olarak korunuyor.
2. EXP-05 smoothing alternatifleri ekranda vardı, CSV'de yoktu: `exp05_smoothing_alternatives.csv` eklendi.
3. EXP-06 cumulative, leave-out ve model AUC tabloları kaydedilmiyordu: ilgili `_auc.csv` dosyaları eklendi.
4. EXP-06 son RF temiz/karışık eğitim karşılaştırması export hücresinden sonra çalışıyordu: export en sona taşındı; RF AUC ve oracle-miss dosyaları eklendi.
5. EXP-06 dört ortamın özellik dağılımları ve ayrı AUC tabloları tamamlandı; `exp06_feature_values.csv` eklendi.
6. EXP-07 dört feature tablosu ve dört test özetinin ayrı export'ları eklendi.
7. EXP-07 salt gürültü satırlarında `balanced_acc`, negatif sınıf doğruluğu olarak yazılıyordu. Ortak metrik fonksiyonu düzeltildi; tek sınıfta bu metrik artık tanımsız (NaN). CSV'deki yedi satır da düzeltildi. FAR ve karışıklık sayıları değişmedi.
8. EXP-07 context etiketinde gerçek sinyal kapsamı yanlış yazılıyordu: k=5, frame=30/hop=10 için 110 yerine 130 ms; k=10 için 210 yerine 230 ms. Kaynak ve metin çıktıları düzeltildi.

## Sonuçların geçerliliğini sınırlayan sorunlar

- EXP-01–06, istenen `Aynursusuz/noisy-speech-dataset` yerine `guynich/librispeech_asr_test_vad` kullanıyor. Bunlar referans etiketli yardımcı deneyler; istenen veri seti üzerindeki deneyler olarak anlatılmamalı.
- EXP-07 `noisy_speech` dosyalarını kullanmıyor; temiz ve salt gürültüden kontrollü SNR karışımları oluşturuyor. Gerçek gürültülü kayıtları ayrıca test etmeden onların başarısı hakkında sonuç verilemez.
- EXP-06 oracle FAR kıyasını düzeltmek, sadece başlığı değiştirmek değildir: bağımsız development konuşmacıları ayırıp eşikleri orada seçerek modelleri yeniden eğitmek gerekir. Mevcut sonuçlar oracle etiketiyle korunuyor.
- Babble negatifleri başka konuşmacıların konuşmasını içeriyor. Hedef “herhangi bir insan konuşması” ise bu negatifler uygun değildir; mevcut sonuçlar foreground konuşmanın etkinliğini ölçüyor. Görev tanımı netleşmeden bu satırlar klasik VAD accuracy'si diye sunulmamalı.
- EXP-07'nin otomatik enerji referansları elle işaretlenmiş konuşma sınırları değildir. Enerji özellikleri aynı varsayımdan üretildiği için değerlendirme yanlılığı olabilir.
- EXP-04 son robust kuralı test hataları incelendikten sonra eklenmiş keşif sonucu. Yeni, dokunulmamış test olmadan bağımsız nihai doğrulama olarak sunulmamalı.
- Kadın/erkek, Türkçe/İngilizce ve ayrı ses etkinliği testleri bu notebook'larla tamamlanmış değil. Çok konuşmacılı deney cinsiyet deneyinin yerine geçmez.
- `exp04_final_results.csv` ve `exp06_feature_comparison.csv`, mevcut notebook'ların export ettiği sonuçlar değil. Eski dosyalardır; yeni tablolarla aynı protokolmüş gibi karşılaştırılmamalı.

## Güvenle söylenebilecek sonuç

Enerji eşiği zayıf konuşmayı kaçırabilir ve yüksek gürültüyü konuşma kabul edebilir. Özellik kombinasyonunun katkısı model ve ortama bağlıdır. Gürültü çeşitliliği ile eğitim yararlı olabilir, fakat kayıt seviyesi değişimi ayrı bir genelleme sorunudur. Nihai özellik/parametre seçimi, bağımsız dev eşikleri ve elle doğrulanmış test etiketleriyle yapılmalıdır.
