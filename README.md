# Retrieval Yöntemlerinin Performans Karşılaştırması

![image](https://github.com/user-attachments/assets/87125c2e-c89f-41e4-94c3-4b296b8ee241)

# Metin/WikiRAG-TR Veri Kümesi:

WikiRAG-TR is a dataset of 5999 question and answer pairs which synthetically created from introduction part of Turkish Wikipedia Articles. The dataset is created to be used for Turkish Retrieval-Augmented Generation (RAG) tasks.
![image](https://github.com/user-attachments/assets/0fa9234a-2f4c-459c-8c9e-d9508a09fdc9)

Dataset Columns<br>
id: Unique identifier for each row.<br>
question: The question generated by the model.<br>
answer: The answer generated by the model.<br>
context: The augmented context containing both relevant and irrelevant information.<br>
is_negative_response: Indicates whether the answer is a negative response (0: No, 1: Yes).<br>
number_of_articles: The number of article introductions used to create the context.<br>
ctx_split_points: The ending character indices of each introduction in the context. These can be used to split the context column into its individual article introductions.<br>
correct_intro_idx: Index of the related introduction in the context. Can be used together with ctx_split_points to find the related introduction. This can also be useful for post-training analysis.<br>

# Performance Result:

Soruların Kelime Uzunluk Dağılımı:
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/soru_uzunluk.png" width="auto">
<br/>

Model Bazlı Chunk İşlem Süresi:
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/chunk_islem_suresi.png" width="auto">
<br/>

Chunk Uzunluklarının Dağılımı:
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/chunk_uzunluk_dagilimi.png" width="auto">
<br/>

Modellerin Karşılaştırılması TOP-1 Accuracy
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/top1_accuracy.png" width="auto">
<br/>

Modellerin Karşılaştırılması TOP-5 Accuracy
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/top5_accuracy.png" width="auto">
<br/>

Modellerin Karşılaştırılması TOP-1 vs TOP-5 Accuracy
<br/>
<img src="https://github.com/raifakyol/LLM_WikiRAG-TR/blob/main/result/top1vstop5_accuracu.png" width="auto">
<br/>

Modellerin Karşılaştırılması TOP-1 vs TOP-5 Accuracy HEATMAP
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
