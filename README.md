# MetaSketch：A Pioneering End-to-End Learning Paradigm Outperforming Handcrafted Methods in Data Stream Sketching
Meta-sketch: A neural data structure for estimating item frequencies of data streams. (AAAI23 Oral)

Learning to Sketch: A Neural Approach to Item Frequency Estimation in Streaming Data. (TPAMI24)


## What is MetaSketch?
We pioneer the use of self-supervised memory neural networks for data stream compression, also known as sketching. Our results demonstrate that **this end-to-end learned compression paradigms** can offer greater potential than traditional **handcrafted methods**. In particular, MetaSketch achieves notably lower error rates under constrained memory budgets.
## Limitations
It is important to note that end-to-end learned data stream compression algorithms are still in their early stages and face certain limitations. For instance, the memory scalability required for real-world deployment may lead to frequent retraining. Nonetheless, we believe that ongoing research will continue to address these challenges and advance the end-to-end learning paradigm. For instance, our latest work [LegoSketch](https://github.com/FFY0/LegoSketch_ICML) (ICML25) effectively resolves the scalability issue and also further reduces compression error.

<img src="https://github.com/FFY0/LegoSketch_ICML/raw/main/Asserts/summarization.png" alt="Error comparision" width="50%">
