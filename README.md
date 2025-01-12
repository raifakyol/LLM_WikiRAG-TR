# Retrieval Yöntemlerinin Performans Karşılaştırması
<h6>Raif AKYOL, 23501086, 2025.</h6>
<h6><a href="https://rakyol.dev/">rakyol.dev</a></h6>

![image](https://github.com/user-attachments/assets/87125c2e-c89f-41e4-94c3-4b296b8ee241)

# Metin/WikiRAG-TR Veri Kümesi:

WikiRAG-TR is a dataset of 5999 question and answer pairs which synthetically created from introduction part of Turkish Wikipedia Articles. The dataset is created to be used for Turkish Retrieval-Augmented Generation (RAG) tasks.
![image](https://github.com/user-attachments/assets/0fa9234a-2f4c-459c-8c9e-d9508a09fdc9)

<h4>Dataset Columns</h4>
<b>id:</b> Unique identifier for each row.<br>
<b>question:</b> The question generated by the model.<br>
<b>answer:</b> The answer generated by the model.<br>
<b>context:</b> The augmented context containing both relevant and irrelevant information.<br>
<b>is_negative_response:</b> Indicates whether the answer is a negative response (0: No, 1: Yes).<br>
<b>number_of_articles:</b> The number of article introductions used to create the context.<br>
<b>ctx_split_points:</b> The ending character indices of each introduction in the context. These can be used to split the context column into its individual article introductions.<br>
<b>correct_intro_idx:</b> Index of the related introduction in the context. Can be used together with ctx_split_points to find the related introduction. This can also be useful for post-training analysis.<br>

# Performance Result:

<h4>Soruların Kelime Uzunluk Dağılımı:</h4>
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/soru_uzunluk.png" width="auto">
<br/>

<h4>Model Bazlı Chunk İşlem Süresi:</h4>
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/chunk_islem_suresi.png" width="auto">
<br/>

<h4>Chunk Uzunluklarının Dağılımı:</h4>
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/chunk_uzunluk_dagilimi.png" width="auto">
<br/>

<h4>Modellerin Karşılaştırılması TOP-1 Accuracy</h4>
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/top1_accuracy.png" width="auto">
<br/>

<h4>Modellerin Karşılaştırılması TOP-5 Accuracy</h4>
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/top5_accuracy.png" width="auto">
<br/>

<h4>Modellerin Karşılaştırılması TOP-1 vs TOP-5 Accuracy</h4>
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/top1vstop5_accuracu.png" width="auto">
<br/>

<h4>Modellerin Karşılaştırılması TOP-1 vs TOP-5 Accuracy HEATMAP</h4>
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/accuracy_heatmap.png" width="auto">
<br/>

                                 Method  Top-1 Accuracy  Top-5 Accuracy
                                   BM25           0.551           0.785
                                   Word           0.254           0.440
                      all-MiniLM-L12-v2           0.203           0.321
         multilingual-e5-large-instruct           0.690           0.912
                        turkish-colbert           0.153           0.300
                              gte-large           0.535           0.776
                                 bge-m3           0.696           0.915
                   Weighted Combination           0.144           0.198


<h4>Sonuçlar:</h4><br/>
Deneyler, farklı bilgi alma yöntemlerinin performanslarını karşılaştırmak için etkili bir temel oluşturmuştur:<br/>
BM25, hızlı ve etkili bir temel yöntem olarak ön plana çıkmıştır.<br/>
Dense modeller, özellikle intfloat/multilingual-e5-large-instruct ve BAAI/bge-m3, yüksek doğruluk sonuçlarıyla dikkat çekmiştir.<br/>
Weighted Combination yönteminin ağırlıklarının iyileştirilmesi gerektiği açıkça görülmüştür.<br/>
Bu sonuçlar, Türkçe dilinde bilgi alma görevleri için yöntemlerin avantaj ve dezavantajlarını değerlendirmek adına önemli bir katkı sağlamaktadır.<br/>

<h4>Kaynakça:</h4><br/>
[1]	WikiRAG-TR, https://huggingface.co/datasets/Metin/WikiRAG-TR, E.T. Ocak 2025.<br/>
[2]	S. Robertson, H. Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond, 2009, pp 333-389.<br/>

