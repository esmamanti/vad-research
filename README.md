# VAD araştırması: kodu ve sonuçları anlama rehberi

Bu proje, kısa ses pencerelerinde konuşma olup olmadığını inceler. Bu rehber mevcut kod ve kaydedilmiş CSV sonuçlarının incelemesidir; İlk incelemede yeni eğitim yapılmamıştır. Sonraki deney denetiminde EXP-01–06 yerel önbellekten yeniden çalıştırılmıştır; EXP-07 modelleri yeniden eğitilmemiştir. Ayrıntılar [deney denetimi raporunda](results/EXPERIMENT_AUDIT.md).

## 1. Önce araştırma sorusunu doğru tanımla

**Ses etkinliği (audio activity):** Mikrofon sinyalinde belirlediğimiz taban seviyesinin üzerinde bir ses var mı? Konuşma, fan, müzik ve kapı sesi buna dahil olabilir. Dijital sessizlik ile sessiz bir odanın mikrofon kaydı aynı şey değildir.

**Konuşma etkinliği (speech activity / VAD):** Seste insan konuşması var mı? Fan sesi yüksek olsa bile konuşma sayılmaz; düşük sesli bir kelime konuşma sayılır.

| Durum | Ses etkinliği | Konuşma etkinliği |
| --- | --- | --- |
| Dijital sıfır | Yok | Yok |
| Taban seviyesinin üzerinde fan sesi | Var | Yok |
| Temiz konuşma | Var | Var |
| Gürültü içindeki konuşma | Var | Var |

Şu an `src/vad_core.py` ve `src/vad_data.py` esas olarak **konuşma / konuşma değil** görevini yapıyor. Ayrı, etiketlenmiş bir ses etkinliği deneyi bulunmuyor. İki görevin etiketleri ve başarı tabloları ayrı olmalı. Ses etkinliği için kullanılan eşik, zayıf konuşmayı önceden elememeli.

## 2. Bir frame nasıl karara dönüşüyor?

Akış: kayıt → mono ve 16 kHz → frame → özellik vektörü → modelin konuşma skoru → karar eşiği → zaman içinde düzeltme → konuşma aralıkları.

- 16 kHz: saniyede 16.000 örnek.
- 30 ms frame: 480 örnekten özellik hesaplanır.
- 10 ms hop: sonraki pencere 160 örnek ileride başlar; pencereler örtüşür.
- Frame boyu, kararın kaç ms'de bir üretildiği değildir. Karar aralığını hop belirler.
- Başlangıç deneyi için 30/10 ms uygundur; optimum olduğunu söylemek için 20/30/40 ms'yi aynı protokolde karşılaştırmak gerekir.
- Büyük frame daha uzun bir sinyali özetler; kısa konuşma başlangıçlarını ve duraklamaları aynı pencerede karıştırabilir.

`extract_features` her frame için **19 özellik** çıkarır: 2 enerji + 1 ZCR + 4 spektral özellik + 12 MFCC. EXP-06 notebook'u ise farklı bir çıkarıcıyla 20 özellik kullanır; `band_ratio` ve `harmonicity` içerir, mutlak enerji içermez. İki deneyin tabloları doğrudan aynı deneymiş gibi karşılaştırılmamalı.

## 3. Özelliklerin anlamı ve değerleri

| Özellik | Hesap / birim | Karara katkısı | Tek başına neden yetmez? |
| --- | --- | --- | --- |
| RMS, `rms_db` | RMS = sqrt(mean(x²)); 20 log10(RMS + epsilon) | Frame'in sinyal seviyesini ölçer | Gürültü de yüksek olabilir; mikrofon seviyesi değişir |
| `rms_rel` | Kırpılmış frame dB'si − kaydın tahmini taban dB'si | Tabanın ne kadar üstünde olduğunu ölçer | Sürekli konuşma veya değişken gürültü taban tahminini bozar |
| ZCR | İşaret değiştiren komşu örneklerin oranı, 0–1 | Dalganın hızlı değişimini gösterir | /s/, /ş/ gibi konuşma sesleri de gürültüye benzer |
| Centroid | Genlik spektrumunun ağırlıklı ortalama frekansı, Hz | Spektrumun ağırlığının düşük/yüksek frekansta oluşu | Konuşmacı, ünsüz ve gürültü türüyle değişir |
| Rolloff | Toplam spektral genliğin %85'ine ulaşılan frekans, Hz | Spektral yayılımı özetler | Evrensel konuşma sınırı yoktur; bu kod güç yerine genlik kullanır |
| Flatness | Güç spektrumunun geometrik ortalaması / aritmetik ortalaması | Düz spektrum ile sivri spektrumu ayırmaya yardım eder | Sessizlikte epsilon davranışı etkiler; konuşma da gürültü benzeri olabilir |
| Flux | Normalize spektrumun önceki frame'e göre değişimi | Zamansal değişimi gösterir | Kapı vurması gibi olaylar da hızla değişir |
| MFCC 1–12 | Mel bant güçleri → log → DCT | Spektral zarfın sıkıştırılmış özeti | Katsayılar için evrensel konuşma eşikleri yoktur |
| Band ratio (EXP-06) | 300–3000 Hz band gücü / toplam güç | Seçilen banttaki yoğunluk | Bazı ortam sesleri de bu banttadır |
| Harmonicity (EXP-06) | Belirli gecikmelerde normalize otokorelasyon tepe değeri | Periyodik, ötümlü seslere ilişkin ipucu | Ötümsüz ünsüzleri kaçırabilir; tonal gürültüler de periyodiktir |

Örnek: frame −35 dB, tahmini taban −55 dB ise göreli enerji yaklaşık 20 dB'dir. Bu, **konuşma kanıtı değil**, tabanın üzerinde etkinlik ipucudur. Koddaki dB dijital genlikten hesaplanır; kalibre edilmiş dB SPL değildir.

“ZCR kaç olursa konuşmadır?” için tek sayı ezberlemek yerine konuşma ve konuşma olmayan frame'lerin dağılımlarını karşılaştır. Her özellik için her ortamda medyan, %10/%90 yüzdelikleri ve histogram çıkar. Dağılımlar örtüşüyorsa o özellik tek başına ayıramaz. AUC ayırma gücünü özetler; ham özellik değerini vermez. Mevcut CSV'lerde sınıfa göre bu ham değer özetleri bulunmuyor.

## 4. Metrikleri okuyabilmek

Pozitif sınıf **konuşma**:

- TP: konuşmayı yakaladı. FN: konuşmayı kaçırdı; kırpmada kelime kaybı riski.
- TN: konuşma olmayanı doğru reddetti. FP: gürültüyü/sessizliği konuşma sandı.
- Accuracy = (TP + TN) / tüm frame'ler.
- Recall = TP / (TP + FN). Miss rate = 1 − recall.
- False alarm rate (FAR) = FP / (FP + TN).
- Precision = TP / (TP + FP). FAR ile aynı değildir.
- F1, precision ve recall'un harmonik ortalamasıdır.
- Balanced accuracy = ((1 − miss) + (1 − FAR)) / 2; iki sınıfa eşit ağırlık verir.
- ROC-AUC, skorların sınıfları farklı eşiklerde ayırma gücüdür. **AUC 0,94 demek %94 accuracy demek değildir.**

Örneğin kaydın %90'ı sessizlikse her frame'e “konuşma yok” diyen model %90 accuracy alır, konuşma recall'u sıfırdır. Kırpma uygulamasında özellikle miss/FAR ikilisine bak.

## 5. Mevcut feature deneyi ne söylüyor?

Kaynak: `results/exp07_feature_ablation.csv`. Aşağıdakiler Random Forest'ın **development** sonuçlarıdır; nihai test başarısı değildir. Eşik aynı development setinde %5 FAR için seçilir. Yüzdeler yuvarlanmıştır.

| Özellik kümesi | ROC-AUC | %5 FAR'da konuşma kaçırma |
| --- | --- | --- |
| Mutlak enerji | 0,800 | %62,71 |
| Göreli enerji | 0,893 | %45,41 |
| Göreli enerji + ZCR | 0,892 | %46,90 |
| Göreli enerji + ZCR + spektral | 0,913 | %39,79 |
| Göreli enerji + ZCR + spektral + MFCC | 0,927 | %29,79 |
| Tüm 19 özellik | 0,939 | %21,26 |

Savunabileceğin sonuç: **Bu veri ve modelde spektral özellikler ile MFCC birlikte fayda sağlamış; ZCR eklemek tek başına iyileştirmemiş.** Daha çok özellik her zaman daha iyi değildir. Bu tablo hangi tek MFCC'nin gerekli olduğunu göstermez; bunun için ekle/çıkar deneyi gerekir. RF'nin feature importance değeri de nedensel katkı kanıtı değildir; ilişkili özellikler önemi paylaşabilir.

## 6. Ortam değişince sonuç değişiyor

Kaynak: `results/exp07_condition_matrix.csv`. Bunlar kaydedilmiş **test** sonuçlarıdır. RF, clean only ve RF, noise-aware aynı tüm özellik grubunu kullanır. Her modelin eşiği development setinde seçilir; testte %5 FAR tutması garanti değildir.

| Test koşulu | Eğitim | Balanced accuracy | Miss | FAR |
| --- | --- | --- | --- | --- |
| Temiz | RF, temiz eğitim | %90,08 | %19,85 | %0,00 |
| Temiz | RF, gürültü eklenmiş eğitim | %95,92 | %7,91 | %0,25 |
| SNR 10 dB | RF, temiz eğitim | %80,15 | %38,89 | %0,81 |
| SNR 10 dB | RF, gürültü eklenmiş eğitim | %91,16 | %17,56 | %0,13 |
| Beyaz gürültü 10 dB | RF, gürültü eklenmiş eğitim | %79,04 | %41,91 | %0,00 |
| Seviye −20 dB, SNR 10 dB | RF, gürültü eklenmiş eğitim | %53,23 | %93,55 | %0,00 |
| Seviye −20 dB, SNR 10 dB | RF, aynı eğitim, mutlak enerji çıkarılmış | %89,85 | %20,13 | %0,17 |

**Önemli yorum:** Normal seviyede tüm özellikler iyi görünürken, düşük kayıt seviyesinde mutlak enerjiyi çıkarmak ciddi iyileşme sağlamış. Dolayısıyla hedef yalnız ortalama skoru artırmak değil; mikrofon seviyesi değişince de çalışmaktır. Gain augmentation sonraki deney için güçlü bir adaydır; burada faydası henüz ölçülmemiştir.

SNR = 10 log10(konuşma gücü / gürültü gücü). 10 dB'de oran 10, 0 dB'de 1'dir. Kayıt seviyesi ve SNR farklı şeylerdir: konuşma ve gürültüyü birlikte kısınca SNR aynı kalabilir.

## 7. Tablolardaki ve yöntemdeki sınırlamalar

1. `noise only` satırlarında pozitif sınıf yoktur. Miss/recall tanımsızdır. İlk sürümde `vad_metrics`, `nanmean` nedeniyle bu satırlarda `balanced_acc` olarak yalnız negatif sınıf doğruluğunu döndürüyordu. Deney denetiminde bu düzeltildi: tek sınıfta balanced accuracy artık NaN. Bu değeri iki sınıflı balanced accuracy ile aynı başlıkta karşılaştırma; salt gürültüde FAR raporla.
2. EXP-06'nın `miss_at_far` fonksiyonu test skorlarından eşik seçiyor. Bunlar test üzerinde hedef FAR'a ayarlanmış, oracle niteliğinde karşılaştırmalardır; sabit development eşiğiyle alınmış nihai test sonucu olarak sunulamaz.
3. EXP-05'in `oracle` satırları da deploy edilebilir ayar olarak sunulmamalı. Frame boylarını eski deneylerin farklı protokolleriyle karşılaştırıp tek optimum ilan etme.
4. `energy_labels` temiz konuşmaya enerji eşiği uyguluyor, kısa boşlukları dolduruyor. Bu **otomatik referans**, elle işaretlenmiş ground truth değildir. Zayıf ünsüzleri yanlış etiketleyebilir; enerji özelliklerinin lehine yanlılık oluşturabilir. Karışımın SNR'sini bilmek konuşma sınırlarını kesin bilmek anlamına gelmez.
5. Eklenen sessizlikler sentetik oda tonu; gerçek konuşma arasındaki bütün duraklamaları temsil etmez. Gürültülü konuşma dosyaları mevcut loader'da kullanılmıyor; temiz konuşma ile salt gürültü karıştırılıyor.
6. Dosya düzeyinde train/dev ayrımı var; EXP-07'de konuşmacı kimliğine göre ayrım garanti edilmemiş. Aynı konuşmacının farklı kayıtları iki tarafa düşerse yeni konuşmacıya genelleme olduğundan iyi görünebilir.
7. `rms_rel` tabanı bütün kayıt üzerinden hesaplanıyor. Gelecek frame'leri kullanan context ve merkezli smoothing de var. Bu sonuçlar mevcut haliyle çevrimdışı işleme içindir; canlı VAD için gecikme ve geçmişe dayalı taban tahmini ayrıca değerlendirilmelidir.
8. 5 komşu frame context'in sinyal kapsamı `(2k × hop) + frame` = 130 ms'dir; notebook etiketi 110 ms yazıyor. 21 skorun ortalaması için merkezler 200 ms yayılır, kapsanan ham sinyal 230 ms'dir. Bu sayılar gecikme ile aynı değildir.
9. Eski `exp06_feature_comparison.csv` mevcut EXP-06 notebook'unun dışa aktardığı dosyalar arasında yok; eski sonuç olarak ayır. Eski/yeni tabloların veri, etiket ve metrik tanımları eşitlenmeden birlikte sıralama yapma.
10. Standart hata/güven aralığı ve birden fazla seed yok. Örtüşen frame'ler bağımsız örnekler değildir; güven aralığını kayıt veya konuşmacı düzeyinde bootstrap ile hesaplamak daha uygundur.

## 8. İstenen sorulara cevap

**1 — Konuşma varken kırpıyorsa?** Bu FN'dir. Eşiği development setinde düşürmek recall'u artırabilir, FAR da artabilir. Ön tampon, hangover ve iki eşikli hysteresis denenebilir. Örneğin başlangıç için 50–100 ms tampon, bitiş için 100–200 ms uzatma birer adaydır. Özellikle kısa/zayıf kelimeler üzerindeki kazanç ile eklenen gürültü süresini ölç; bunlar evrensel optimum değildir.

**2 — Parametreleri ekleyip çıkarınca?** Aynı kayıt ayrımı, aynı referans, aynı ortam ve aynı değerlendirme ile tek değişkeni değiştir. Enerji → enerji+ZCR → spektral → MFCC ekleme ve tüm kümeden tek grubu çıkarma deneyi yap. Özelliğin faydasını yalnız eğitim skoru üzerinden seçme.

**3 — Ortam sesleriyle eğitim?** Mevcut sonuçlarda temiz eğitime göre bazı koşullarda açık iyileşme var. Fakat beyaz gürültü ve düşük kayıt seviyesi gibi koşullarda zayıflık sürüyor. Farklı gürültüler, SNR'ler, gain ve oda etkilerini eğitimde çeşitlendir; ayrıca hiç görülmemiş ortamlarla test et.

**4 — Kadın/erkek sesi?** Sesin temel frekansı ve spektral yapısı konuşmacılar arasında değişebilir; cinsiyet tek açıklayıcı değildir. Bu repodan kadın/erkek başarı farkı için güvenilir sonuç çıkmıyor. Metadata veya gönüllü beyanıyla gruplar belirlenmeli; birden fazla konuşmacı, benzer ortam/SNR ve konuşmacıdan bağımsız ayrım ile grup recall/FAR'ları ölçülmeli. Sesten cinsiyet tahmini yapıp gerçek metadata yerine koyma.

**5 — İngilizce eğitim, Türkçe test?** VAD kelimeyi tanımaya çalışmaz; akustik örüntülerden karar verir. Bu yüzden diller arasında çalışması mümkündür, ancak bu projede Türkçe başarısı ölçülmemiştir. Dil farkını mikrofon/ortam farkından ayıran eşleştirilmiş Türkçe test gerekir. Kendi Türkçe kaydın pilot olabilir; tek konuşmacıyla genel dil sonucu çıkmaz.

**6 — Farklı ortamda eğitim?** Eğitim ortamı × test ortamı matrisi kur. Mevcut EXP-07 esas olarak temiz eğitim ve gürültü eklenmiş eğitim kıyaslıyor; her ayrı ortamda ayrı model eğitilmiş tam bir matris değil. Evrensel doğruluk sayısı verilemez.

**7 — Veri seti?** Veri kartındaki `speech/noisy` sınıfları doğrudan VAD etiketleri değildir. `noisy`, hem gürültülü konuşmayı hem salt gürültüyü içerir. Dosya kategorilerini ayırmak gerekiyor. Frame etiketleri için elle sınır veya doğrulanmış temiz/gürültülü eşleşme gerekir. Kaynak: https://huggingface.co/datasets/Aynursusuz/noisy-speech-dataset

**8 — Kendi kayıt?** Kendi konuşmanı, sessizlikler ve salt ortam sesi bölümleriyle kaydet. Başlangıç/bitişleri dinleyerek elle işaretle; sonra modelle karşılaştır. Modelin kendi çıktısını referans etiket olarak kullanma. Kayıt ve değerlendirme henüz tamamlanmış değil.

**9 — Feature/frame seçimi?** 30 ms / 10 ms başlangıç; göreli enerji, ZCR, spektral özellikler ve MFCC karşılaştırması. Mutlak enerjiyi hem dahil ederek hem çıkararak, gain değişimleri altında test et. Ham özellik değerlerinin sınıf/ortam dağılımları da rapora girmeli.

**10 — Derin öğrenme?** Geleneksel deney protokolü sabitlendikten sonra aynı train/dev/test üzerinde log-Mel girdili küçük CNN/CRNN ve hazır VAD baseline kıyaslanabilir. Başarıya ek olarak gecikme, CPU süresi ve model boyutu raporlanmalı. Bu aşama mevcut repoda tamamlanmış değil.

## 9. Sıradaki somut deney

1. **Tanım ve etiket:** Ses etkinliği ile konuşma etkinliğini ayrı tanımla. Bir pilot Türkçe kayıt için iki ayrı zaman aralığı etiket listesi oluştur. Konuşma için zayıf ünsüzleri koruyan elle işaretleme yap.
2. **Adil ayrım:** Konuşmacı/kaynak kayıt ve aynı kayıttan üretilmiş tüm varyantlar aynı split'te kalsın. Train model eğitimi, dev özellik/eşik seçimi, test tek nihai değerlendirme için kullanılsın.
3. **Feature değerleri:** Her ortam/sınıf/özellik için medyan, p10, p90; histogram ve sınıf örtüşmesini kaydet. Ölçekleme yalnız train'e fit edilsin; LR mevcut pipeline bunu yapıyor.
4. **Kontrollü tarama:** Önce sabit frame ile feature ablation; sonra seçilen birkaç kümede 20/30/40 ms ve 5/10/20 ms hop. LR ve RF'yi karşılaştır. Her kombinasyonu testte seçme.
5. **Robustluk:** Eğitimde gain değişimini ve farklı gürültüleri ayrı deney olarak ekle. Testte temiz, SNR 20/10/5/0, salt gürültü ve görülmeyen ortamları raporla.
6. **Karar:** Dev üzerinde eşik–miss–FAR eğrisi çıkar. %5 FAR hedefiyle sınırlı kalma; konuşmayı koruma hedefi için kabul edilebilir miss oranını ve maliyetini tanımla. Tampon/hangover sonrası FAR'ı yeniden ölç.
7. **Rapor:** Her satırda split, model, feature kümesi, frame/hop, eşik kaynağı, ortam, TP/TN/FP/FN, recall, FAR, F1 ve iki sınıf varsa balanced accuracy olsun. Salt gürültü satırında FAR öncelikli olsun.

## 10. Hocaya anlatabileceğin kısa açıklama

“İki görevi ayırıyorum: ses etkinliği ve konuşma etkinliği. Enerji sesin seviyesini ölçüyor, fakat gürültü de enerjik olduğu için konuşmayı tek başına ayıramıyor. Bu nedenle spektral özellikler ve MFCC ekleyerek kontrollü karşılaştırma yapıyorum. Mevcut development sonuçlarında RF ile tüm özellikler AUC 0,939 veriyor; bu %93,9 accuracy anlamına gelmiyor. Testte gürültü eklenmiş eğitim SNR 10 dB'de balanced accuracy'yi %80,15'ten %91,16'ya çıkarıyor. Ancak seviye 20 dB düşünce aynı model konuşmanın %93,55'ini kaçırıyor. Mutlak enerjiyi çıkarınca bu %20,13'e iniyor. Bu nedenle özellikleri yalnız ortalama başarıyla değil, mikrofon ve ortam değişimine dayanıklılık üzerinden seçiyorum. Şu an referanslar enerji tabanlı; kendi Türkçe kaydımı elle etiketleyerek bağımsız doğrulama yapmam gerekiyor.”
