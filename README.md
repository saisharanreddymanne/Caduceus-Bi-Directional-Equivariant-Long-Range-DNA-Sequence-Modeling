# Caduceus : Bi-Directional Equivariant Long-Range DNA Sequence Modeling

**Abstract**

Large-scale sequence modeling has catalyzed signif
icant advancements in biology and genomics, enabling deeper
 insights into complex DNA interactions. However, genomic se
quence modeling presents unique challenges, such as the need to
 capture long-range dependencies, the influence of upstream and
 downstream regions, and reverse complementarity (RC) of DNA.
 Building upon these challenges, the Caduceus model was intro
duced as the first family of RC-equivariant bi-directional long
range DNA language models. Leveraging the MambaDNA block
 for RC-equivariance and bi-directionality, Caduceus achieves
 state-of-the-art performance on variant effect prediction tasks
 and outperforms models ten times its size that lack these
 capabilities.
 While Caduceus demonstrates exceptional potential, its scal
ability and attention mechanisms present opportunities for en
hancement. In this work, we propose novel extensions, including
 BigBird, GNN+Attention, and Hybrid CNN Transformers, to
 address these limitations. These models leverage sparse attention,
 graph neural networks, and hybrid convolutional-transformer
 architectures to achieve superior accuracy, precision, recall, and
 F1-scores on genomic benchmarks such as enhancer prediction
 and long-range sequence analysis. Our results showcase that Big
Bird excels in handling long-range dependencies, GNN+Attention
 captures complex sequence relationships, and the Hybrid CNN
 Transformer combines local pattern recognition with global
 contextual understanding. This study sets a foundation for
 scalable and efficient DNA sequence modeling, paving the way
 for advancements in genomic research.
 

**INTRODUCTION**

 The advent of large-scale sequence modeling has ushered
 in groundbreaking advancements not only in natural language
 processing (NLP) but also in biology, genomics, and medicine.
 In proteomics, these models have facilitated remarkable break
throughs, such as the accurate prediction of protein structures,
 decoding amino acid interactions, and designing new thera
peutic molecules. Similarly, genomic sequence modeling has
 become a pivotal tool for understanding the intricate behaviors
 of DNA, including gene mutations, interactions, and regulatory
 elements. This has significantly advanced our ability to explore
 cellular mechanisms at a molecular level. However, genomic
 DNA, unlike proteins, contains both coding and non-coding
 regions, with the latter playing a crucial role in regulating cel
lular activities and influencing phenotypic outcomes. Modeling
 these non-coding regions requires architectures capable of
 recognizing both local and global patterns across genomic se
quences. Additionally, DNA’s long-range dependencies, where
 genomic regions far apart on the same strand can interact
 and affect cellular functions, and the reverse complementarity
 (RC) between the two DNA strands, add layers of complexity
 to sequence modeling. The need to model both strands of
 DNA and account for their complementary information is a
 significant hurdle in predictive genomic analysis. Furthermore,
 tasks such as Variant Effect Prediction (VEP), which aims to
 determine the influence of genetic mutations on phenotypes,
 require efficient modeling of long-range interactions spanning
 millions of base pairs.
 To address these challenges, the Caduceus model was
 introduced, offering a solution through its innovative use of
 RC-equivariance and bi-directionality. By incorporating the
 MambaDNA block, Caduceus facilitates the handling of long
 genomic sequences while accounting for bidirectional context
 and RC-equivariance, properties essential for accurate genomic
 predictions. This model overcomes the computational inef
f
 iciencies typically associated with attention-based architec
tures, enabling it to scale to large genomic datasets. Caduceus
 has demonstrated remarkable performance, surpassing the ca
pabilities of models that do not incorporate RC-equivariance
 or bi-directionality, even when compared to models that are
 up to 10 times larger in scale.
 However, despite Caduceus’s successes, it faces challenges
 in terms of scalability and computational efficiency, particu
larly when applied to complex, large-scale genomic datasets.
 To address these limitations, this research extends the Ca
duceus framework by integrating advanced architectures like
 BigBird, GNN+Attention, and Hybrid CNN Transformers.
Each of these models brings unique advantages:- BigBird
 utilizes sparse attention mechanisms, allowing it to efficiently
 model long-range dependencies without the computational
 cost of traditional attention models, making it particularly
 effective for large genomic datasets.- GNN+Attention com
bines the capabilities of Graph Neural Networks (GNNs) with
 attention mechanisms to capture the complex relationships and
 dependencies between distant genomic elements, enabling the
 model to learn intricate patterns in the data.- Hybrid CNN
 Transformers combine convolutional neural networks (CNNs)
 for local pattern recognition with transformers for modeling
 global dependencies, striking a balance between capturing
 f
 ine-grained motifs and understanding broad contextual rela
tionships in DNA sequences.
 These extended models not only improve accuracy but
 also enhance other performance metrics, such as precision,
 recall, and F1-score, across various benchmark genomic tasks,
 including enhancer prediction, variant effect prediction, and
 long-range sequence analysis. The integration of these ad
vanced architectures offers scalable and efficient solutions
 for analyzing complex genomic data, paving the way for
 significant breakthroughs in personalized medicine, mutation
 analysis, and genomic regulatory element prediction.

 **EXPERIMENTS AND RESULTS**
 
 **A. Re produced Result**
 
 The table provides a comprehensive comparison of four
 model architectures—Base Paper (Caduceus), BigBird RC,
 GNN + Attention, and the Hybrid CNN Transformer—based
 on various aspects of their design, capabilities, and per
formance. Each model employs a unique approach to se
quence representation. The Base Paper model uses raw se
quences, while BigBird RC utilizes k-mer embeddings (3
mers). GNN+Attention employs graph-based representations,
 and the Hybrid CNN Transformer builds on hierarchical k
mer embeddings, showcasing advanced handling of sequence
 structure.
 In terms of attention mechanisms, the Base Paper relies
 on basic attention, whereas BigBird RC employs sparse at
tention to process long sequences efficiently. GNN+Attention
 introduces spatial mechanisms alongside traditional attention,
 and the Hybrid CNN Transformer integrates Transformer
based self-attention for robust performance. The embedding
 layers further distinguish the models: the Base Paper uses
 basic token embeddings, BigBird RC benefits from pre-trained
 embeddings, GNN+Attention adopts node and edge embed
dings, and the Hybrid CNN Transformer incorporates fine
tuned embeddings for greater adaptability.

<img width="262" height="68" alt="image" src="https://github.com/user-attachments/assets/f154973f-9b76-4755-a3eb-e740e71fa175" />


**Table 1: MODEL COMPARISON IN TERMS OF PERFORMANCE METRICS**
 
 Handling long sequences varies significantly across these
 architectures. The Base Paper is limited to 512 tokens, while
 BigBird RC supports sequences up to 131,000 tokens, lever
aging its sparse attention mechanism. GNN+Attention han
dles moderately long sequences constrained by graph size,
 and the Hybrid CNN Transformer processes long sequences
 efficiently using hierarchical pooling techniques. Addition
ally, only BigBird RC, GNN+Attention, and Hybrid CNN
 Transformer address class imbalance through methods such as
 weighted sampling, graph augmentation, and GAN-augmented
 sequences, respectively.

 <img width="251" height="55" alt="image" src="https://github.com/user-attachments/assets/a0f4f78a-490c-4e64-89d6-498f783fc5b2" />

 **Fig. 3. Result Comparison Table**

 
 Synthetic data is absent in the Base Paper but is integrated
 into the other models. BigBird RC utilizes GAN-generated se
quences, while GNN+Attention and Hybrid CNN Transformer
 employ augmented graph and sequence data, respectively.
 Training optimization strategies differ as well, with the Base
 Paper employing a simple loss function (e.g., cross-entropy).
 In contrast, BigBird RC uses optimized sparse Transformer
 training, GNN+Attention leverages graph-specific optimizers,
 and the Hybrid CNN Transformer incorporates stratified loss
 functions.
 Performance metrics highlight that BigBird RC achieves the
 highest accuracy (91.2%), precision (91.0%), recall (90.8%),
 and F1-score (91.0%). GNN+Attention and Hybrid CNN
 Transformer also deliver strong results, with accuracies of
 89.8% and 89.2%, respectively. The Base Paper lags be
hind with 87.5% across all key metrics. Computational ef
f
 iciency varies as well: the Base Paper and GNN+Attention
 exhibit moderate efficiency due to their attention mecha
nisms and graph complexity, whereas BigBird RC and Hybrid
 CNNTransformer achieve higher efficiency through optimized
 sparse attention and CNN layers.
 This comparison highlights the trade-offs between model
 architectures in terms of performance, sequence handling, and
 computational efficiency. BigBird RC excels in processing
 long sequences and achieving high accuracy, while the Hy
brid CNN Transformer balances computational efficiency with
 robust performance, making it a versatile option for complex
 sequence tasks.
