# CRISPR-MTL: Multi-Task Learning with DNABERT for Joint gRNA On-Target and Off-Target Prediction

---

## Daftar Isi

1. [Abstrak](#abstrak)
2. [Latar Belakang](#latar-belakang)
3. [Urgensi dan Skala Masalah](#urgensi-dan-skala-masalah)
4. [Tinjauan Literatur dan Gap Penelitian](#tinjauan-literatur-dan-gap-penelitian)
5. [Rumusan Masalah dan Tujuan Penelitian](#rumusan-masalah-dan-tujuan-penelitian)
6. [Novelty dan Kontribusi](#novelty-dan-kontribusi)
7. [Metodologi](#metodologi)
8. [Rancangan Eksperimen](#rancangan-eksperimen)
9. [Hasil yang Diharapkan](#hasil-yang-diharapkan)
10. [Rencana Implementasi 3 Hari](#rencana-implementasi-3-hari)
11. [Keterbatasan dan Risiko](#keterbatasan-dan-risiko)
12. [Referensi Kunci](#referensi-kunci)

---

## Abstrak

Teknologi CRISPR-Cas9 telah merevolusi biologi molekuler dengan memungkinkan pengeditan genom secara presisi, namun dua tantangan fundamental tetap menghambat adopsi klinisnya: prediksi efisiensi pemotongan di lokasi target yang diinginkan (on-target efficiency) dan prediksi pemotongan tidak disengaja di lokasi lain yang memiliki urutan serupa (off-target activity). Hingga saat ini, kedua masalah ini hampir selalu dimodelkan secara terpisah menggunakan model deep learning independen, meskipun secara biologis keduanya mencerminkan fenomena yang sama, yaitu seberapa kuat dan spesifik ikatan antara gRNA dengan DNA.

Penelitian ini mengusulkan **CRISPR-MTL**, sebuah kerangka multi-task learning yang mengadaptasi **DNABERT** sebagai shared encoder pretrained untuk mempelajari representasi sekuens gRNA secara bersama-sama, kemudian menggunakan dua prediction head terpisah untuk masing-masing tugas prediksi. DNABERT dipilih karena merupakan model transformer yang sudah di-pretrain pada miliaran pasang basa DNA dari berbagai organisme, sehingga representasinya jauh lebih kaya dibanding encoder yang dilatih dari nol pada dataset CRISPR yang kecil. Selain model terlatih dan laporan teknis, penelitian ini juga menyertakan prototipe aplikasi berbasis Gradio yang memungkinkan pengguna memprediksi skor on-target dan off-target secara interaktif dari sekuens gRNA yang diinput.

Hipotesis utama penelitian ini adalah bahwa fine-tuning shared DNA foundation model secara simultan pada dua tugas yang berkorelasi secara biologis akan menghasilkan representasi yang lebih informatif dibanding dua model independen. Kontribusi tambahan berupa analisis komparatif Integrated Gradients antara kedua prediction head untuk mengidentifikasi apakah kedua tugas berfokus pada posisi nukleotida yang sama atau berbeda dalam sekuens gRNA.

---

## Latar Belakang

### Apa itu CRISPR-Cas9?

CRISPR-Cas9 (Clustered Regularly Interspaced Short Palindromic Repeats) adalah sistem pengeditan genom yang diadaptasi dari mekanisme pertahanan alami bakteri terhadap serangan virus. Sistem ini terdiri dari dua komponen utama: protein **Cas9** yang berfungsi sebagai "gunting" molekuler, dan **guide RNA (gRNA)** yang berfungsi sebagai pemandu untuk membawa Cas9 ke lokasi DNA yang tepat. Setelah menemukan lokasi target, Cas9 memotong kedua untai DNA (double-strand break), dan sel kemudian memperbaiki potongan tersebut melalui mekanisme yang dapat dimanfaatkan untuk menyisipkan, menghapus, atau mengganti urutan genetik tertentu.

Teknologi ini telah membuka era baru dalam penelitian biomedis, termasuk pengembangan terapi gen untuk penyakit genetik langka, pengobatan kanker berbasis pengeditan sel imun, dan riset fungsional genomik skala besar. Potensi klinisnya yang luar biasa inilah yang membuat masalah keamanan menjadi sangat krusial: kesalahan sekecil apapun dalam pengeditan dapat berdampak fatal bagi pasien yang sedang menjalani terapi.

### Apa itu Guide RNA (gRNA)?

Guide RNA adalah rantai RNA sintetik yang dirancang oleh ilmuwan, biasanya sepanjang sekitar 20 nukleotida (dalam konteks ini, beserta 3 nukleotida tambahan yang disebut PAM sequence "NGG", totalnya menjadi 23-mer). Urutan nukleotida gRNA dirancang agar komplementer dengan urutan DNA target yang ingin dipotong. Komposisi spesifik nukleotida dalam gRNA sangat mempengaruhi dua hal yang saling terkait: seberapa efisien pemotongan di lokasi target (on-target efficiency) dan seberapa besar risiko pemotongan di lokasi lain yang tidak diinginkan (off-target activity). Hubungan antara urutan nukleotida dan dua properti ini sangat kompleks dan nonlinear, yang menjadikannya masalah yang cocok untuk pendekatan deep learning.

### On-Target Efficiency vs Off-Target Activity

**On-target efficiency** mengukur seberapa baik dan konsisten Cas9 berhasil memotong tepat di lokasi DNA yang menjadi target. Diukur sebagai indel frequency, yaitu persentase sel dalam populasi yang mengalami pemotongan di lokasi target setelah gRNA diintroduksikan. Nilai ini berkisar antara 0 hingga 1, di mana nilai tinggi menunjukkan gRNA yang efisien.

**Off-target activity** mengukur seberapa sering Cas9 secara tidak sengaja memotong di lokasi lain yang hanya mirip tetapi tidak identik dengan urutan target. Dalam DNA manusia dengan 3 miliar pasang basa, ada ratusan hingga ribuan lokasi yang memiliki urutan yang berbeda hanya pada 1, 2, atau 3 posisi (disebut mismatch). Jika Cas9 memotong di lokasi-lokasi ini, gen yang seharusnya berfungsi normal dapat rusak, dan dalam kasus terburuk dapat memicu onkogenesis.

Korelasi biologis antara keduanya adalah kunci dari seluruh riset ini. Faktor-faktor yang membuat sebuah gRNA kuat dalam berikatan dengan DNA target, seperti konten GC yang tinggi, minimnya struktur sekunder, dan stabilitas termodinamika ikatan gRNA-DNA, adalah faktor yang sama yang membuat gRNA berisiko berikatan dengan off-target sites. Korelasi biologis ini adalah fondasi ilmiah dari hipotesis bahwa shared encoder akan lebih informatif dari dua encoder yang dilatih secara terpisah.

---

## Urgensi dan Skala Masalah

Bagian ini menyajikan data kuantitatif untuk menunjukkan seberapa besar dan mendesak masalah yang ingin diselesaikan oleh CRISPR-MTL.

### Skala Pertumbuhan Uji Klinis CRISPR

Per Oktober 2024, terdapat lebih dari 100 studi terdaftar di ClinicalTrials.gov yang melibatkan teknologi CRISPR, mencakup 106 uji klinis pengeditan gen, 16 uji klinis base editing, dan 1 uji klinis prime editing. Dua terapi CRISPR pertama di dunia, yaitu Casgevy untuk penyakit sel sabit dan beta-thalassemia, telah mendapat persetujuan FDA pada Desember 2023, menandai babak baru adopsi klinis yang akan terus mendorong pertumbuhan jumlah uji klinis secara eksponensial dalam beberapa tahun ke depan.

Secara geografis, Amerika Serikat dan China memimpin jumlah uji klinis CRISPR, diikuti oleh Uni Eropa. Fase uji yang sedang berjalan mencakup phase I, II, dan III untuk berbagai kondisi termasuk kanker, penyakit hematologi, penyakit kardiovaskular, dan penyakit infeksi.

### Risiko Off-Target sebagai Hambatan Utama Adopsi Klinis

Penerapan CRISPR dalam terapi manusia menghadapi satu hambatan regulasi yang paling konsisten: bukti komprehensif bahwa tidak ada pengeditan yang tidak disengaja di lokasi selain target. Off-target effects pada CRISPR dapat mencakup indel kecil, variasi struktural, translokasi kromosom, inversi, dan delesi besar, yang semuanya berpotensi menjadi sumber risiko genotoksisitas bagi pasien.

Sebagai ilustrasi nyata, sebuah studi menemukan bahwa analisis off-target untuk gRNA yang sedang diuji dalam uji klinis penyakit sel sabit dan beta-thalassemia melewatkan satu off-target site yang signifikan: site tersebut diintroduksikan oleh sebuah varian genetik dengan frekuensi alel minor 4.5% pada populasi keturunan Afrika, artinya hampir 1 dari 20 individu dari kelompok populasi tersebut memiliki risiko yang tidak terprediksi oleh analisis standar.

Lebih jauh lagi, pada tahun 2025 terjadi kematian pertama yang terkait dengan uji klinis CRISPR, meskipun penyebabnya adalah respons imun terhadap vektor pengiriman (AAV6) bukan off-target editing itu sendiri. Kejadian ini memperkuat urgensi bahwa setiap aspek keamanan terapi CRISPR, termasuk prediksi off-target, harus diverifikasi secara menyeluruh sebelum administrasi pada manusia.

### Biaya dan Inefisiensi Verifikasi Eksperimental

Proses verifikasi off-target secara eksperimental menggunakan teknik seperti GUIDE-seq, CIRCLE-seq, atau SITE-seq membutuhkan infrastruktur laboratorium khusus, memakan waktu berminggu-minggu, dan biayanya dapat mencapai puluhan ribu dolar per kandidat gRNA ketika diperhitungkan biaya reagen, sequencing, dan analisis. Dalam konteks pengembangan terapi gen di mana ratusan kandidat gRNA perlu disaring sebelum satu kandidat terpilih untuk diuji lebih lanjut, biaya dan waktu yang dibutuhkan untuk verifikasi eksperimental menjadi bottleneck yang sangat signifikan.

Di sinilah model komputasional seperti CRISPR-MTL memiliki peran yang jelas: menyaring kandidat gRNA secara in silico terlebih dahulu untuk mengeliminasi kandidat berisiko tinggi sebelum verifikasi eksperimental yang mahal dilakukan. Bahkan jika model hanya mampu mengeliminasi 30-40% kandidat berisiko tinggi dengan presisi yang cukup, penghematan waktu dan biaya dalam pipeline pengembangan terapi gen sudah sangat signifikan.

### Konteks Regulasi

FDA, EMA, dan berbagai badan regulasi internasional lainnya mensyaratkan analisis off-target yang komprehensif sebagai bagian dari paket pengajuan Investigational New Drug (IND) application sebelum uji klinis dapat dimulai. Pedoman terbaru dari FDA menekankan bahwa prediksi komputasional dan verifikasi eksperimental harus digunakan secara komplementer, bukan salah satu saja. Ini secara langsung memposisikan model prediksi komputasional seperti CRISPR-MTL sebagai komponen yang relevan secara regulasi, bukan hanya alat penelitian akademik semata.

---

## Tinjauan Literatur dan Gap Penelitian

### Evolusi Model untuk On-Target Prediction

Penelitian komputasional untuk prediksi efisiensi on-target telah berkembang dalam tiga generasi. Generasi pertama menggunakan pendekatan berbasis aturan dan skor heuristik. Generasi kedua dimulai oleh **Azimuth** (Doench et al., 2016) yang menggunakan gradient boosting machine dengan fitur yang direkayasa secara manual, mencapai Spearman correlation sekitar 0.55 hingga 0.65. Kemudian **DeepHF** (Wang et al., 2019) memperkenalkan LSTM untuk representasi sekuens otomatis. Puncak dari era ini adalah **CRISPRon** (Xiang et al., 2021, Nature Communications) yang dilatih pada 23.902 gRNA dan menunjukkan performa yang secara signifikan lebih tinggi dari tools sebelumnya pada empat dataset test independen, menjadikannya SOTA on-target saat ini.

### Evolusi Model untuk Off-Target Prediction

**CRISPR-Net** (Lin and Wong, 2020) memperkenalkan encoding scheme 7x23 untuk merepresentasikan berbagai tipe mismatch dan menggunakan arsitektur Inception-BiLSTM. **R-CRISPR** (2021) memperbaiki robustness untuk dataset dengan insersi dan delesi. Pada 2024, dua paper signifikan muncul: **CRISPR-M** (Sun et al., 2024, PLoS Computational Biology) mengusulkan multi-view encoding dengan CNN dan BiLSTM, sementara **CRISPR-DIPOFF** (Toufikuzzaman et al., 2024, Briefings in Bioinformatics) menjadi model pertama yang menggabungkan Integrated Gradients untuk interpretasi biologis dari prediksi off-target, menemukan dua sub-region penting dalam seed region. Untuk off-target berbasis BERT, **CrisprBERT** (Sari et al., 2024, Bioinformatics Advances) mengadaptasi BERT dengan BiLSTM embedding dan terbukti mengungguli model konvensional pada dataset CHANGE-seq.

### SOTA Terkini: Foundation Model untuk DNA

**DNABERT** (Ji et al., 2021) adalah model pertama yang mengadaptasi arsitektur BERT untuk sekuens DNA menggunakan k-mer tokenization dan pretraining pada seluruh genom manusia. Paper terbaru (2025, PLoS ONE) menunjukkan bahwa fine-tuning DNABERT dengan proses dua tahap untuk off-target prediction menghasilkan performa yang mengungguli semua baseline termasuk CRISPR-DIPOFF dan CrisprBERT. **CCLMoff** (Du et al., 2025, Communications Biology) menggunakan RNA language model dari RNAcentral dan mencapai AUROC=0.985 pada dataset DIG-seq, menjadikannya model off-target terkuat yang dilaporkan saat ini. **CRISMER** (2025, biorxiv) mengusulkan arsitektur CNN-Transformer yang dapat mengoptimasi kandidat gRNA secara langsung.

### Peta SOTA Ringkas

| Model | Tahun | Task | Arsitektur | Performa |
|---|---|---|---|---|
| Azimuth | 2016 | On-target | Gradient Boosting | Spearman ~0.55-0.65 |
| DeepHF | 2019 | On-target | LSTM | Spearman ~0.87 (cross-val) |
| CRISPRon | 2021 | On-target | CNN-LSTM | SOTA on-target (indep. test) |
| CRISPR-Net | 2020 | Off-target | Inception-BiLSTM | AUROC ~0.80-0.85 |
| CRISPR-M | 2024 | Off-target | Multi-view CNN+BiLSTM | Di atas CRISPR-Net |
| CRISPR-DIPOFF | 2024 | Off-target | LSTM + Integrated Gradients | SOTA interpretable |
| CrisprBERT | 2024 | Off-target | BERT + BiLSTM | Di atas model konvensional |
| DNABERT fine-tuned | 2025 | Off-target | DNABERT | Di atas CRISPR-DIPOFF & CrisprBERT |
| CCLMoff | 2025 | Off-target | RNA Language Model | AUROC=0.985 (DIG-seq) |
| CRISMER | 2025 | Off-target | CNN-Transformer | SOTA + optimasi gRNA |

### Gap yang Belum Dieksplorasi

Meskipun kemajuan yang ada sangat signifikan, dua gap kritis belum ditangani oleh satupun paper di atas.

**Gap pertama** adalah pemisahan artifisial antara on-target dan off-target modeling. Seluruh model di atas membangun model terpisah untuk masing-masing tugas. Tidak ada yang mengeksplorasi apakah keduanya dapat dan seharusnya dimodelkan bersama dalam satu kerangka multi-task, padahal justifikasi biologisnya sudah sangat kuat.

**Gap kedua** adalah analisis komparatif interpretabilitas lintas tugas. CRISPR-DIPOFF sudah menggunakan Integrated Gradients, tetapi hanya untuk satu tugas. Belum ada studi yang membandingkan posisi nukleotida yang dianggap penting antara on-target head dan off-target head dalam satu model yang sama.

---

## Rumusan Masalah dan Tujuan Penelitian

**Rumusan masalah utama:** Apakah fine-tuning DNABERT sebagai shared encoder dalam kerangka multi-task learning untuk memprediksi on-target efficiency dan off-target activity secara simultan menghasilkan performa yang lebih baik dibandingkan model single-task yang masing-masing dilatih secara independen?

Tujuan pertama adalah membangun kerangka multi-task learning yang mengadaptasi DNABERT sebagai shared encoder pretrained, dengan dua prediction head untuk on-target efficiency (regression) dan off-target cleavage activity (classification) secara simultan.

Tujuan kedua adalah membuktikan melalui ablation study yang sistematis bahwa penggunaan DNABERT lebih baik dari encoder dari scratch, dan bahwa framework multi-task lebih baik dari dua model DNABERT single-task independen.

Tujuan ketiga adalah menghasilkan analisis komparatif Integrated Gradients antara kedua head untuk mengidentifikasi apakah on-target head dan off-target head berfokus pada posisi sekuens yang serupa atau berbeda, dan menginterpretasikan temuan tersebut terhadap pengetahuan biologis tentang seed region.

Tujuan keempat adalah mengimplementasikan prototipe aplikasi prediksi berbasis Gradio yang memungkinkan pengguna memasukkan sekuens gRNA secara interaktif dan mendapatkan prediksi skor on-target dan off-target beserta visualisasinya.

---

## Novelty dan Kontribusi

**Kontribusi pertama: Adaptasi pertama DNABERT untuk multi-task CRISPR prediction.** Seluruh studi sebelumnya menggunakan DNABERT atau foundation model DNA lainnya hanya untuk satu tugas. CRISPR-MTL adalah, sepengetahuan penulis, adaptasi pertama DNABERT dalam kerangka multi-task learning yang secara simultan memproses on-target dan off-target prediction dalam satu model bersama.

**Kontribusi kedua: Validasi empiris korelasi biologis on-target dan off-target.** Korelasi biologis antara dua tugas ini sudah diketahui secara konseptual, namun belum pernah divalidasi secara empiris melalui shared representation learning. Jika model multi-task terbukti lebih baik dari dua model independen, ini adalah bukti komputasional pertama bahwa representasi sekuens yang informatif untuk on-target memang juga informatif untuk off-target.

**Kontribusi ketiga: Analisis komparatif Integrated Gradients lintas tugas.** Berbeda dari CRISPR-DIPOFF yang hanya menganalisis saliency untuk satu tugas, penelitian ini membandingkan peta kepentingan posisi nukleotida antara dua head dalam satu model yang sama, memberikan wawasan baru tentang apakah dua tugas bergantung pada fitur yang tumpang tindih atau komplementer.

**Kontribusi keempat: Prototipe aplikasi end-to-end yang dapat digunakan.** Integrasi model ke dalam aplikasi Gradio menjadikan CRISPR-MTL bukan sekadar eksperimen akademik, melainkan alat yang dapat langsung dicoba oleh peneliti biologi untuk pre-screening kandidat gRNA sebelum validasi eksperimental.

---

## Metodologi

### Dataset

**Dataset On-Target (untuk Head 1):**

Dataset utama yang digunakan adalah **Doench 2016 dataset** (Doench et al., Nature Biotechnology, 2016). Dataset ini berisi 5.310 guide sequence dengan label indel frequency hasil eksperimen high-throughput menggunakan plasmid library yang diekspresikan dalam sel manusia, dan telah menjadi standar benchmark di bidang ini selama hampir satu dekade. Sebagai dataset pelengkap, data dari **DeepHF** (Wang et al., 2019) juga diintegrasikan untuk memperbesar corpus training menjadi sekitar 8.000 hingga 10.000 sampel berlabel.

Sumber download: Dataset Doench 2016 tersedia di `https://github.com/khaled-buet/CRISPRpred`. Dataset DeepHF tersedia di repository GitHub resmi DeepHF.

**Dataset Off-Target (untuk Head 2):**

Dataset off-target yang digunakan berasal dari repository `https://github.com/dagrate/public_data_crisprCas9` yang mengumpulkan data benchmark dalam format yang sudah di-encode, mencakup data dari GUIDE-seq (Listgarten et al.), CIRCLE-seq (Tsai et al.), SITE-seq (Cameron et al.), dan Digenome-seq. Setiap sampel adalah pasangan gRNA-DNA dengan label biner (1 jika terjadi off-target cleavage, 0 jika tidak). Dataset ini secara inheren sangat imbalanced dan ditangani melalui class weighting dalam fungsi loss. Kedua dataset tidak memiliki sampel yang overlap, sehingga ditangani melalui strategi alternating batch training.

### Model SoTA yang Diadaptasi: DNABERT

**Apa itu DNABERT?**

DNABERT (Ji et al., 2021, *Bioinformatics*) adalah model transformer berbasis arsitektur BERT yang di-pretrain pada sekuens DNA genomik menggunakan tokenisasi 6-mer. Dengan vocabulary berukuran 4^6 = 4.096 kemungkinan hexanucleotide, model ini mampu menangkap pola kontekstual dalam sekuens DNA yang jauh lebih kaya dibanding one-hot encoding konvensional. DNABERT di-pretrain menggunakan Masked Language Modeling (MLM) pada seluruh genom manusia (GRCh38), menghasilkan model dengan 12 layer transformer, 768 dimensi hidden, dan 12 attention heads (sekitar 110 juta parameter). Model tersedia di HuggingFace dengan identifier `zhihan1996/DNA_bert_6`.

**Justifikasi pemilihan DNABERT:**

Pertama, DNABERT adalah satu-satunya model yang di-pretrain khusus pada DNA genomik dan tersedia publik di HuggingFace, sehingga dapat di-load tanpa pretraining ulang. Kedua, paper terbaru (2025, PLoS ONE) telah memvalidasi secara langsung bahwa fine-tuning DNABERT untuk off-target prediction mengungguli semua baseline termasuk CRISPR-DIPOFF dan CrisprBERT. Ketiga, mekanisme multi-head self-attention pada DNABERT secara natural menangkap dependensi jarak jauh dalam sekuens, termasuk dependensi antara seed region dan PAM site yang secara biologis penting, tanpa rekayasa fitur manual.

### Representasi Input dan Tokenisasi

Untuk sekuens gRNA 23-mer pada on-target prediction, tokenisasi menggunakan jendela 6-mer overlapping menghasilkan 18 token, ditambah [CLS] dan [SEP] menjadi total 20 token input. Contoh konkretnya adalah sebagai berikut:

```
Sekuens gRNA : ATCGGCATCGGATCGGCATCGGG
Token 6-mer  : ATCGGC TCGGCA CGGCAT GGCATC ... (18 token)
Input DNABERT: [CLS] ATCGGC TCGGCA CGGCAT ... TCGGG [SEP]
```

Untuk pasangan gRNA-DNA pada off-target prediction, BERT pair encoding digunakan dengan format `[CLS] gRNA_tokens [SEP] DNA_tokens [SEP]`, menghasilkan total 38 token. Format pair encoding ini memungkinkan mekanisme cross-attention dalam transformer untuk secara alami memodelkan interaksi antara sekuens gRNA dan DNA target.

### Arsitektur Model CRISPR-MTL

```
ARSITEKTUR LENGKAP CRISPR-MTL
================================

INPUT ON-TARGET                    INPUT OFF-TARGET
[CLS] + 18 token + [SEP]           [CLS] + 18 token [SEP] + 18 token [SEP]
(panjang = 20 token)               (panjang = 38 token)
        |                                      |
        +------------------+-------------------+
                           |
                  SHARED ENCODER
                  DNABERT (pretrained)
                  12 Transformer Layers
                  Hidden dim = 768 | 12 Attention Heads
                  Layer 1-8  : FROZEN
                  Layer 9-12 : Fine-tuned
                           |
                  Ambil [CLS] token representation
                  (768-dim)
                           |
                  Dropout(p=0.1)
                  Linear(768 -> 256) + GELU
                  LayerNorm
                  (Projection Layer, dilatih dari nol)
                           |
           +---------------+---------------+
           |                               |
 HEAD 1: ON-TARGET               HEAD 2: OFF-TARGET
 EFFICIENCY (Regression)         CLEAVAGE (Classification)
 Linear(256 -> 64) + ReLU        Linear(256 -> 64) + ReLU
 Dropout(p=0.2)                  Dropout(p=0.3)
 Linear(64 -> 1) + Sigmoid       Linear(64 -> 1) + Sigmoid

 OUTPUT: skor 0-1                OUTPUT: probabilitas
 (prediksi indel frequency)      off-target cleavage
```

Pemilihan representasi **[CLS] token** mengikuti standar BERT asli sebagai agregator informasi dari seluruh sekuens. **Dropout asimetris** diterapkan karena off-target head lebih berisiko overfitting akibat dataset yang lebih kecil dan imbalanced. **Pembekuan layer 1-8** mencegah catastrophic forgetting dari pengetahuan umum tentang DNA sekaligus mempercepat training. **Projection layer** mempelajari transformasi spesifik-tugas dari representasi umum DNABERT.

### Strategi Fine-Tuning Multi-Task

Training dilakukan dengan **alternating batch training** dalam dua fase.

**Fase 1: Warm-up (Epoch 1-5).** Seluruh layer DNABERT dibekukan. Hanya projection layer dan kedua prediction head yang dilatih. Tujuannya adalah membiarkan head menyesuaikan diri dengan distribusi output DNABERT sebelum mulai mengubah representasi internal model.

**Fase 2: Full Fine-Tuning (Epoch 6 seterusnya).** Layer 9-12 DNABERT dibuka dan ikut dioptimasi. Layer 1-8 tetap dibekukan. Learning rate untuk layer DNABERT yang di-fine-tune dibuat 10 kali lebih kecil dari learning rate head (discriminative fine-tuning).

Pada setiap iterasi alternating: ambil batch on-target, hitung L_on, backward dan update layer 9-12 DNABERT + projection + Head 1. Kemudian ambil batch off-target, hitung L_off, backward dan update layer 9-12 DNABERT + projection + Head 2. Ulangi hingga konvergen.

### Fungsi Loss

Loss dioptimasi secara terpisah dalam alternating scheme untuk menghindari masalah skala antar task.

Untuk on-target head menggunakan **MSE**:

```
L_on = (1/N) * sum((y_pred - y_true)^2)
```

Untuk off-target head menggunakan **BCE dengan class weighting**:

```
L_off = -sum( w_pos * y * log(y_pred) + w_neg * (1-y) * log(1-y_pred) )
w_pos = n_negatif / n_positif   (bobot kelas minoritas)
w_neg = 1.0
```

### Interpretability: Integrated Gradients

Setelah model selesai dilatih, Integrated Gradients diterapkan pada embedding layer DNABERT menggunakan library **Captum** dari PyTorch. Karena tokenisasi k-mer yang overlapping, setiap nukleotida asli muncul dalam beberapa token berbeda. Agregasi saliency score dari level token ke level nukleotida dilakukan dengan menjumlahkan skor absolut dari semua token yang mengandung nukleotida tersebut, kemudian dinormalisasi, menghasilkan satu skor importance per posisi nukleotida yang divisualisasikan sebagai bar chart heatmap.

Analisis komparatif dilakukan dengan membandingkan profile kepentingan posisi antara Head 1 dan Head 2. Validasi biologis dilakukan dengan mengecek apakah posisi seed region (posisi 12-20 dari 5' gRNA) secara konsisten muncul sebagai posisi paling penting pada off-target head, sesuai dengan pengetahuan biologis tentang mekanisme Cas9.

### Prototipe Aplikasi: Gradio Interface

Sebagai komponen implementasi software, model CRISPR-MTL yang sudah dilatih diintegrasikan ke dalam antarmuka web sederhana menggunakan **Gradio**. Interface ini memungkinkan siapapun untuk memprediksi on-target efficiency dan off-target risk dari sebuah sekuens gRNA tanpa perlu menjalankan kode Python secara langsung.

Desain interface mencakup tiga komponen utama. Pertama, sebuah text input field di mana pengguna memasukkan sekuens gRNA 23-mer (dengan validasi otomatis bahwa input hanya mengandung karakter A, T, G, C dan panjangnya tepat 23). Kedua, dua output gauge atau progress bar yang menampilkan on-target efficiency score (0.0 hingga 1.0) dan off-target risk score (0.0 hingga 1.0) beserta interpretasi verbal (misalnya "High Efficiency", "Low Risk"). Ketiga, sebuah bar chart yang menampilkan saliency map per posisi nukleotida untuk memberikan penjelasan tentang posisi mana yang paling mempengaruhi prediksi.

Contoh kode implementasi Gradio yang akan digunakan:

```python
import gradio as gr
import torch
import matplotlib.pyplot as plt

def predict_grna(sequence):
    # Validasi input
    sequence = sequence.upper().strip()
    if len(sequence) != 23 or not all(c in "ATGC" for c in sequence):
        return "Error: Masukkan sekuens gRNA 23-mer yang valid (hanya A, T, G, C)", None, None

    # Tokenisasi dan prediksi
    kmer_seq = " ".join([sequence[i:i+6] for i in range(len(sequence)-6+1)])
    
    # On-target prediction
    on_score = model_predict_ontarget(kmer_seq)
    
    # Off-target prediction (menggunakan sekuens gRNA sebagai referensi diri)
    off_score = model_predict_offtarget(kmer_seq)
    
    # Saliency map
    saliency = compute_integrated_gradients(kmer_seq)
    fig = plot_saliency(saliency, sequence)

    return (
        f"On-Target Efficiency: {on_score:.3f}",
        f"Off-Target Risk: {off_score:.3f}",
        fig
    )

interface = gr.Interface(
    fn=predict_grna,
    inputs=gr.Textbox(
        label="Masukkan Sekuens gRNA (23-mer)",
        placeholder="Contoh: ATCGGCATCGGATCGGCATCGGG"
    ),
    outputs=[
        gr.Textbox(label="On-Target Efficiency Score"),
        gr.Textbox(label="Off-Target Risk Score"),
        gr.Plot(label="Saliency Map per Posisi Nukleotida")
    ],
    title="CRISPR-MTL: gRNA Predictor",
    description="Masukkan sekuens gRNA Cas9 23-mer untuk memprediksi efisiensi on-target dan risiko off-target secara simultan."
)

interface.launch()
```

---

## Rancangan Eksperimen

Seluruh eksperimen diorganisasi dalam tiga kelompok utama ditambah satu analisis post-hoc, dengan total **8 training run** dan **1 analisis interpretability**. Semua run menggunakan 5-fold cross-validation dengan stratified sampling.

---

### Kelompok 1: Baseline Experiments (Hari 1)

Tujuan kelompok ini adalah menyediakan angka pembanding yang solid sebelum model utama dilatih. Ada empat run dalam kelompok ini, dibagi menjadi dua sub-kelompok berdasarkan jenis encoder yang digunakan.

#### Exp-A1: Single-Task BiLSTM dari Scratch (On-Target)

```
Tujuan    : baseline on-target tanpa pretrained model, sebagai lower bound
Input     : one-hot encoding gRNA 23-mer → shape (23, 4)
Arsitektur: Linear(4→32) → BiLSTM(hidden=128, layers=2, bidirectional=True)
            → ambil hidden state terakhir → Linear(256→64) → ReLU
            → Linear(64→1) → Sigmoid
Task      : Regression
Loss      : MSE
Dataset   : Doench 2016 + DeepHF (~8.000-10.000 sampel)
Split     : 5-fold cross-validation (80% train, 20% val per fold)
Optimizer : Adam, lr=1e-3
Epochs    : 50 (dengan early stopping patience=10)
Output    : Spearman correlation, Pearson correlation (mean ± std across folds)
Estimasi  : ~5-10 menit per fold di GPU T4/P100
```

#### Exp-A2: Single-Task CNN-BiLSTM dari Scratch (Off-Target)

```
Tujuan    : baseline off-target tanpa pretrained model, replikasi pendekatan CRISPR-Net
Input     : mismatch matrix gRNA-DNA pair → shape (23, 7)
            7 channel: 4 one-hot gRNA + tipe mismatch (match/sub/ins/del)
Arsitektur: Conv1D(in=7, out=64, kernel=3) → ReLU → MaxPool
            → BiLSTM(hidden=128, bidirectional=True)
            → Linear(256→64) → ReLU → Dropout(0.3)
            → Linear(64→1) → Sigmoid
Task      : Binary Classification
Loss      : BCE dengan class weighting (w_pos = n_neg/n_pos)
Dataset   : GUIDE-seq + CIRCLE-seq + SITE-seq + Digenome-seq
Split     : 5-fold stratified cross-validation
Optimizer : Adam, lr=1e-3
Epochs    : 50 (dengan early stopping patience=10)
Output    : AUROC, AUPR (mean ± std across folds)
Estimasi  : ~5-10 menit per fold di GPU T4/P100
```

#### Exp-B1: DNABERT Single-Task (On-Target)

```
Tujuan    : baseline on-target dengan DNABERT, untuk mengisolasi kontribusi multi-task
Input     : k-mer tokenized gRNA (6-mer overlapping) → [CLS] + 18 token + [SEP] = 20 token
Arsitektur: DNABERT pretrained (zhihan1996/DNA_bert_6)
            Layer 1-8 : FROZEN
            Layer 9-12: fine-tuned
            Ambil [CLS] representation (768-dim)
            → Dropout(0.1) → Linear(768→256) → GELU → LayerNorm (projection)
            → Linear(256→64) → ReLU → Dropout(0.2) → Linear(64→1) → Sigmoid
Task      : Regression
Loss      : MSE
Dataset   : sama dengan A1
Split     : 5-fold cross-validation
Optimizer : AdamW dengan discriminative LR
            LR projection + head = 1e-4
            LR DNABERT layer 9-12 = 1e-5
Epochs    : 30 (DNABERT konvergen lebih cepat dari scratch)
Output    : Spearman correlation, Pearson correlation (mean ± std across folds)
Estimasi  : ~15-25 menit per fold di GPU T4/P100
```

#### Exp-B2: DNABERT Single-Task (Off-Target)

```
Tujuan    : baseline off-target dengan DNABERT, untuk mengisolasi kontribusi multi-task
Input     : k-mer pair encoding → [CLS] + 18 gRNA token + [SEP] + 18 DNA token + [SEP] = 38 token
Arsitektur: DNABERT pretrained (zhihan1996/DNA_bert_6)
            Layer 1-8 : FROZEN
            Layer 9-12: fine-tuned
            Ambil [CLS] representation (768-dim)
            → Dropout(0.1) → Linear(768→256) → GELU → LayerNorm (projection)
            → Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
Task      : Binary Classification
Loss      : BCE dengan class weighting
Dataset   : sama dengan A2
Split     : 5-fold stratified cross-validation
Optimizer : AdamW dengan discriminative LR
            LR projection + head = 1e-4
            LR DNABERT layer 9-12 = 1e-5
Epochs    : 30
Output    : AUROC, AUPR (mean ± std across folds)
Estimasi  : ~15-25 menit per fold di GPU T4/P100
```

---

### Kelompok 2: Main Model (Hari 2)

Ini adalah eksperimen inti yang menjadi klaim utama novelty penelitian.

#### Exp-MTL-Full: CRISPR-MTL Full Model

```
Tujuan    : model utama yang diklaim, membuktikan hipotesis shared representation
Input     : dua jenis input yang di-batch secara bergantian
            Batch on-target : [CLS] + 18 gRNA token + [SEP] (panjang 20)
            Batch off-target: [CLS] + 18 gRNA token + [SEP] + 18 DNA token + [SEP] (panjang 38)

Arsitektur: DNABERT shared encoder (zhihan1996/DNA_bert_6)
            Layer 1-8 : FROZEN selama Fase 1 dan 2
            Layer 9-12: frozen selama Fase 1, fine-tuned selama Fase 2
            → [CLS] token (768-dim)
            → Dropout(0.1) → Linear(768→256) → GELU → LayerNorm (shared projection)
            ┌─ Head 1 (On-Target) : Linear(256→64) → ReLU → Dropout(0.2) → Linear(64→1) → Sigmoid
            └─ Head 2 (Off-Target): Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid

Training  :
  Fase 1 (Epoch 1-5, Warm-up):
    - Semua DNABERT layer FROZEN
    - Hanya projection layer + Head 1 + Head 2 yang dilatih
    - LR = 1e-4 untuk semua yang tidak dibekukan
    - Alternating: 1 batch on-target (update projection + Head 1)
                   1 batch off-target (update projection + Head 2)

  Fase 2 (Epoch 6-30, Full Fine-Tuning):
    - Layer 9-12 DNABERT di-unfreeze
    - Discriminative LR:
        LR DNABERT layer 9-12 = 1e-5
        LR projection + heads = 1e-4
    - Alternating training tetap sama seperti Fase 1
    - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)

Loss      :
    Saat batch on-target  : L_on  = MSE(pred_on, true_on)
    Saat batch off-target : L_off = BCE_weighted(pred_off, true_off)
    Tidak ada combined loss, setiap step hanya mengoptimasi satu loss

Checkpoint: simpan model terbaik berdasarkan rata-rata (norm_spearman + norm_auroc)
            di mana kedua metrik dinormalisasi ke range 0-1 terhadap baseline

Split     : 5-fold cross-validation (fold yang sama dengan Baseline)
Output    : Spearman + Pearson (Head 1), AUROC + AUPR (Head 2)
Estimasi  : ~30-45 menit per fold di GPU T4/P100
```

---

### Kelompok 3: Ablation Study (Hari 2, setelah MTL-Full)

Setiap ablation mengubah tepat satu komponen dari MTL-Full untuk mengisolasi kontribusinya. Semua hyperparameter lain identik dengan MTL-Full.

#### Exp-ABL1: Frozen All Layers

```
Perubahan : semua 12 layer DNABERT dibekukan sepanjang training
            hanya projection layer + Head 1 + Head 2 yang dilatih
Pertanyaan: apakah fine-tuning layer akhir DNABERT memang memberikan perbedaan?
Ekspektasi: performa lebih rendah dari MTL-Full, terutama pada task yang
            distribusi sekuensnya berbeda dari pretraining DNABERT
Estimasi  : ~15-20 menit per fold (lebih cepat karena gradient tidak melewati DNABERT)
```

#### Exp-ABL2: Unfreeze All Layers

```
Perubahan : semua 12 layer DNABERT di-fine-tune dari epoch 6 seterusnya
            (bukan hanya layer 9-12)
LR        : DNABERT layer 1-8 = 1e-6 (sangat kecil)
            DNABERT layer 9-12 = 1e-5
            projection + heads = 1e-4
Pertanyaan: apakah fine-tuning terlalu dalam malah merugikan
            (catastrophic forgetting dari pengetahuan DNA umum)?
Ekspektasi: mungkin lebih buruk dari MTL-Full pada off-target karena dataset kecil
            lebih rentan overfitting ketika encoder berubah terlalu banyak
Estimasi  : ~35-50 menit per fold (gradient melewati semua layer)
```

#### Exp-ABL3: Combined Loss

```
Perubahan : ganti alternating batch training dengan combined loss
            pada setiap step, KEDUA loss dihitung dan dijumlahkan:
            L_total = 0.5 * L_on + 0.5 * L_off
            Kedua dataset di-sample secara bersamaan dalam satu batch gabungan
Pertanyaan: apakah strategi alternating lebih baik dari combined loss?
            apakah perbedaan skala antara MSE dan BCE menjadi masalah?
Ekspektasi: combined loss mungkin tidak stabil karena skala MSE (biasanya 0.01-0.1)
            dan skala BCE (biasanya 0.3-0.7) berbeda jauh, menyebabkan salah satu
            task mendominasi gradient update
Estimasi  : ~30-45 menit per fold
```

---

### Eksperimen Tambahan: Interpretability Analysis (Hari 3)

Ini bukan eksperimen training baru, melainkan analisis post-hoc menggunakan checkpoint terbaik dari MTL-Full.

#### Exp-IG: Integrated Gradients Comparative Analysis

```
Tujuan    : mengidentifikasi posisi nukleotida paling penting menurut
            masing-masing head, dan membandingkan keduanya secara berdampingan

Setup     :
  Model   : checkpoint MTL-Full terbaik (fold dengan performa tertinggi)
  Library : Captum (torch.hub)
  Baseline: embedding vektor nol (zero embedding) untuk semua token

Sampel analisis:
  Untuk Head 1 (On-Target):
    - 10 gRNA dengan prediksi efficiency TINGGI (top 10 dari val set)
    - 10 gRNA dengan prediksi efficiency RENDAH (bottom 10 dari val set)
  Untuk Head 2 (Off-Target):
    - 10 pasangan gRNA-DNA dengan prediksi off-target aktif TINGGI
    - 10 pasangan dengan prediksi off-target aktif RENDAH

Komputasi :
  Jumlah langkah integral : 50 (Riemann approximation)
  Agregasi token ke nukleotida:
    score_posisi_i = sum(|IG_j|) untuk semua token j yang mengandung nukleotida i
    kemudian dinormalisasi sehingga total = 1.0

Output    :
  1. Bar chart saliency per posisi (1-23) untuk Head 1, rata-rata dari 20 sampel
  2. Bar chart saliency per posisi (1-23) untuk Head 2, rata-rata dari 20 sampel
  3. Side-by-side comparison plot antara Head 1 dan Head 2
  4. Highlight otomatis pada seed region (posisi 12-20 dari ujung 5')

Validasi biologis:
  Cek apakah posisi 12-20 (seed region) mendominasi saliency Head 2 (off-target)
  Bandingkan distribusi kepentingan Head 1 vs Head 2:
    - Jika Head 2 lebih terkonsentrasi pada seed region → hipotesis biologis terkonfirmasi
    - Jika keduanya mirip → menunjukkan shared mechanism yang kuat

Estimasi  : ~30-60 menit total (tidak butuh GPU besar, hanya inference + gradient)
```

---

### Ringkasan Semua Run

| ID | Model | Task | Data | Metrik | Estimasi/Fold |
|---|---|---|---|---|---|
| A1 | BiLSTM scratch | On-target | Doench+DeepHF | Spearman, Pearson | 5-10 mnt |
| A2 | CNN-BiLSTM scratch | Off-target | GUIDE-seq+dll | AUROC, AUPR | 5-10 mnt |
| B1 | DNABERT single-task | On-target | Doench+DeepHF | Spearman, Pearson | 15-25 mnt |
| B2 | DNABERT single-task | Off-target | GUIDE-seq+dll | AUROC, AUPR | 15-25 mnt |
| MTL-Full | CRISPR-MTL | Both | Both | Spearman + AUROC/AUPR | 30-45 mnt |
| ABL1 | MTL frozen all | Both | Both | Spearman + AUROC/AUPR | 15-20 mnt |
| ABL2 | MTL unfreeze all | Both | Both | Spearman + AUROC/AUPR | 35-50 mnt |
| ABL3 | MTL combined loss | Both | Both | Spearman + AUROC/AUPR | 30-45 mnt |
| IG | Interpretability | Post-hoc | 40 sampel | Saliency visualization | 30-60 mnt total |

**Total estimasi GPU time:** ~15-20 jam untuk semua 8 run lengkap dengan 5-fold CV di Kaggle P100/Colab T4.

### Hyperparameter Standar Semua Eksperimen

Untuk memastikan perbandingan yang adil, berikut adalah hyperparameter yang dibuat konsisten di seluruh eksperimen kecuali disebutkan sebaliknya.

| Hyperparameter | Nilai |
|---|---|
| Batch size | 32 (on-target), 16 (off-target, karena dataset lebih kecil) |
| Optimizer | AdamW |
| Weight decay | 1e-2 |
| Gradient clipping | max_norm = 1.0 |
| Cross-validation | 5-fold stratified |
| Random seed | 42 (semua eksperimen) |
| Mixed precision | fp16 (aktifkan jika GPU mendukung) |
| Checkpoint metric | rata-rata Spearman + AUROC yang dinormalisasi |

### Metrik Evaluasi

Untuk **on-target head**, metrik utama adalah **Spearman Rank Correlation Coefficient** antara prediksi model dan nilai indel frequency eksperimental. Spearman dipilih karena mengukur korelasi urutan peringkat yang lebih relevan secara praktis: peneliti ingin mengetahui gRNA mana yang relatif lebih efisien. Pearson correlation dilaporkan sebagai metrik sekunder.

Untuk **off-target head**, dua metrik digunakan bersama. **AUROC** mengukur kemampuan model membedakan off-target aktif dari non-aktif pada berbagai threshold. **AUPR** lebih informatif untuk dataset imbalanced karena langsung mengukur performa pada kelas minoritas yang secara klinis paling penting.

### Device Requirements dan Rekomendasi

Karena DNABERT adalah model ~110 juta parameter namun dengan input yang sangat pendek (maksimal 38 token), kebutuhan komputasi jauh lebih ringan dibanding fine-tuning LLM teks biasa.

| Device | Status | Estimasi Total Semua Run |
|---|---|---|
| Kaggle P100 (16GB VRAM) | Direkomendasikan | ~15-20 jam |
| Google Colab T4 (15GB VRAM) | Cukup, risiko disconnect | ~18-25 jam |
| Local GPU RTX 3060+ (8GB VRAM) | Ideal jika tersedia | ~10-15 jam |
| CPU saja | Tidak direkomendasikan | 80+ jam |

Rekomendasi utama adalah **Kaggle** sebagai environment primer karena tidak ada session timeout dan limit 30 jam GPU per minggu sudah lebih dari cukup. Wajib menyimpan checkpoint setelah setiap fold selesai untuk mencegah kehilangan progress.

---

## Hasil yang Diharapkan

Untuk perbandingan DNABERT vs model dari scratch: DNABERT diharapkan memberikan keunggulan yang signifikan pada kedua tugas, karena representasi pretrained yang kaya mempercepat konvergensi dan menghasilkan generalisasi yang lebih baik pada dataset yang relatif kecil.

Untuk perbandingan multi-task vs single-task DNABERT: keunggulan multi-task diharapkan lebih jelas pada off-target head, karena dataset off-target yang lebih kecil akan lebih banyak diuntungkan dari sinyal tambahan yang berasal dari on-target data melalui shared encoder.

Untuk ablation pembekuan layer: fine-tuning layer 9-12 DNABERT diharapkan memberikan keunggulan dibanding membekukan semua layer, karena layer-layer akhir perlu beradaptasi dengan distribusi sekuens gRNA yang spesifik.

Untuk analisis saliency: kedua head diharapkan menunjukkan elevated importance pada seed region (posisi 12-20 dari 5'), namun dengan profil yang berbeda. Off-target head diharapkan lebih terkonsentrasi pada seed region, sementara on-target head menunjukkan distribusi yang lebih merata karena efisiensi on-target dipengaruhi oleh komposisi keseluruhan sekuens.

---

## Rencana Implementasi 3 Hari

### Hari 1: Setup dan Baseline (Target: Angka Baseline Tersedia)

Mulai dengan menginstall dependencies: `transformers`, `torch`, `captum`, `gradio`, `sklearn`, `scipy`, dan `pandas`. Verifikasi bahwa DNABERT dapat di-load dan berfungsi dengan benar:

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
model = BertModel.from_pretrained("zhihan1996/DNA_bert_6")

def seq_to_kmer(seq, k=6):
    return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

grna = "ATCGGCATCGGATCGGCATCGGG"
inputs = tokenizer(seq_to_kmer(grna), return_tensors="pt")
outputs = model(**inputs)
cls_repr = outputs.last_hidden_state[:, 0, :]
print(cls_repr.shape)  # torch.Size([1, 768])
```

Download dan parse kedua dataset. Latih Baseline A1 (BiLSTM dari scratch, on-target) dan A2 (CNN-BiLSTM, off-target) selama 20-30 epoch. Kemudian latih Baseline B1 (DNABERT single-task on-target) dan B2 (DNABERT single-task off-target). Catat semua angka dalam tabel ringkasan di akhir Hari 1.

### Hari 2: Model Utama dan Ablation (Target: CRISPR-MTL Terlatih)

Implementasikan kelas `CRISPRMultiTask` dalam PyTorch yang mengintegrasikan DNABERT, projection layer, dan dua head. Implementasikan alternating training loop dengan dua fase (warm-up 5 epoch dengan semua DNABERT frozen, kemudian full fine-tuning dengan layer 9-12 unfrozen dan discriminative learning rate).

Setelah model utama selesai, jalankan ablation study: frozen all layers, fine-tune all layers, dan combined loss sebagai kontrol. Catat semua hasil dalam tabel terorganisasi dan simpan semua checkpoint terbaik. Di akhir Hari 2, tabel perbandingan lengkap antara semua baseline dan CRISPR-MTL sudah harus tersedia.

**Catatan penting:** Mulai mengerjakan slide pitch deck secara paralel di Hari 2 agar tidak menjadi bottleneck di Hari 3.

### Hari 3: Interpretability, Aplikasi, Laporan, dan Pitch Deck (Target: Semua Deliverable)

Implementasikan Integrated Gradients menggunakan Captum. Hitung saliency untuk 20 sampel gRNA dari masing-masing task dan buat visualisasi heatmap komparatif. Buat prototipe Gradio menggunakan kode template yang sudah ada di bagian Metodologi. Pastikan interface berjalan secara lokal dan dapat didemokan.

Finalisasi laporan teknis dalam format Markdown. Push seluruh kode ke GitHub dengan README yang mencakup:
- Instruksi instalasi dependencies
- Cara menjalankan training (`python train.py`)
- Cara menjalankan evaluasi (`python evaluate.py`)
- Cara menjalankan aplikasi Gradio (`python app.py`)
- Contoh output dari masing-masing skrip

Finalisasi pitch deck 10-12 slide dengan struktur berikut. Slide 1 adalah halaman judul. Slide 2 dan 3 membahas problem statement dengan data kuantitatif dari bagian Urgensi. Slide 4 adalah tinjauan SOTA dan gap yang belum dieksplorasi. Slide 5 menjelaskan pendekatan CRISPR-MTL dengan justifikasinya. Slide 6 menampilkan diagram arsitektur. Slide 7, 8, dan 9 menyajikan hasil eksperimen dalam tabel perbandingan dan grafik ablation. Slide 10 menampilkan visualisasi saliency map komparatif. Slide 11 menampilkan demo screenshot aplikasi Gradio. Slide 12 adalah kesimpulan, kontribusi, dan rencana pengembangan.

---

## Keterbatasan dan Risiko

**Keterbatasan pertama: Dataset yang tidak overlap.** Alternating batch training adalah pendekatan yang lebih lemah dibanding multi-task training dengan dataset yang benar-benar berbagi sampel. Validitas empiris hipotesis shared representation bergantung pada seberapa efektif DNABERT dapat belajar dari dua distribusi data berbeda secara bergantian.

**Keterbatasan kedua: Ukuran dataset off-target yang kecil.** Dataset off-target publik yang tersedia relatif kecil, membatasi kemampuan generalisasi off-target head ke gRNA yang tidak muncul dalam training set.

**Keterbatasan ketiga: Tidak ada validasi eksperimental.** Seluruh klaim performa berbasis perbandingan komputasional pada dataset benchmark. Validasi eksperimental di laboratorium tidak dilakukan dalam scope penelitian ini.

**Keterbatasan keempat: DNABERT vs model DNA yang lebih baru.** Tersedia model DNA yang lebih baru seperti DNABERT-2, Nucleotide Transformer, dan HyenaDNA yang menunjukkan performa superior dibanding DNABERT original. Penelitian ini menggunakan DNABERT original karena keterbatasan waktu dan sumber daya komputasi, dan eksplorasi model yang lebih baru merupakan arah pengembangan yang jelas untuk future work.

**Mitigasi risiko teknis:** Jika alternating training terbukti tidak stabil selama Hari 2, strategi fallback adalah sequential fine-tuning: fine-tune DNABERT pada on-target data terlebih dahulu, kemudian fine-tune pada off-target data dengan sebagian layer dibekukan. Strategi ini tetap memiliki narasi ilmiah yang valid tentang transfer learning antar domain.

---

## Referensi Kunci

**Wajib dibaca sebelum Hari 1 (domain dan dataset):**

Doench, J.G., Fusi, N., Sullender, M., et al. (2016). Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9. *Nature Biotechnology*, 34(2), 184-191.

Tsai, S.Q., Zheng, Z., Nguyen, N.T., et al. (2015). GUIDE-seq enables genome-wide profiling of off-target cleavage by CRISPR-Cas nucleases. *Nature Biotechnology*, 33(2), 187-197.

Davies, K., Philippidis, A., & Barrangou, R. (2024). Five Years of Progress in CRISPR Clinical Trials (2019-2024). *The CRISPR Journal*. (konteks klinis dan data jumlah uji klinis)

**Wajib dibaca untuk model SoTA yang diadaptasi:**

Ji, Y., Zhou, Z., Liu, H., & Davuluri, R.V. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*, 37(15), 2112-2120.

**Referensi untuk baseline model (baca abstract dan tabel hasil):**

Xiang, X., Corsi, G.I., Anthon, C., et al. (2021). Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep learning. *Nature Communications*, 12, 3238. (CRISPRon)

Lin, J. & Wong, K.C. (2020). Off-target predictions in CRISPR-Cas9 gene editing using deep learning. *Bioinformatics*, 36(S2), i724-i732. (CRISPR-Net)

**Referensi SOTA terkini 2024-2025:**

Toufikuzzaman, M., Samee, M.A.H., & Rahman, M.S. (2024). CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction. *Briefings in Bioinformatics*, 25(2), bbad530.

Sun, J., Guo, J., & Liu, J. (2024). CRISPR-M: Predicting sgRNA off-target effect using a multi-view deep learning network. *PLoS Computational Biology*, 20(3), e1011972.

Sari, O., Liu, Z., Pan, Y., & Shao, X. (2024). Predicting CRISPR-Cas9 off-target effects in human primary cells using bidirectional LSTM with BERT embedding. *Bioinformatics Advances*, 5(1), vbae184. (CrisprBERT)

Du, W., et al. (2025). CCLMoff: a versatile CRISPR/Cas9 system off-target prediction tool using language model. *Communications Biology*. (SOTA saat ini, AUROC=0.985 pada DIG-seq)

Luo, Y., et al. (2025). Improved CRISPR/Cas9 off-target prediction with DNABERT. *PLoS ONE*. (validasi empiris DNABERT untuk domain CRISPR)

**Referensi untuk regulasi dan konteks klinis:**

Anzalone, A.V., et al. (2024). Measurement and clinical interpretation of CRISPR off-targets. *Nature Reviews Genetics*. (kerangka regulasi evaluasi off-target)

Cancellieri, S., et al. (2022). Human genetic diversity alters off-target outcomes of therapeutic gene editing. *Nature Genetics*. (bukti kasus off-target dalam uji klinis nyata)

**Referensi untuk interpretability dan metodologi:**

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *Proceedings of ICML 2017*. (paper asli Integrated Gradients)

Kokhlikyan, N., et al. (2020). Captum: A unified and generic model interpretability library for PyTorch. *arXiv:2009.07896*.

Howard, J. & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL 2018*. (referensi untuk discriminative fine-tuning)

---

*Dokumen ini berfungsi sebagai research proposal sekaligus panduan implementasi lengkap untuk proyek CRISPR-MTL. Semua keputusan desain yang tercantum memiliki justifikasi yang dapat dipertahankan secara ilmiah, teknis, dan sesuai dengan kriteria penilaian final project yang mensyaratkan penggunaan model SoTA, implementasi software end-to-end, dan pitch deck yang meyakinkan.*
