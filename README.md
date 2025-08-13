# MetaSketch：A Pioneering End-to-End Learning Paradigm Outperforming Handcrafted Methods in Data Stream Sketching
Meta-sketch: A neural data structure for estimating item frequencies of data streams. (AAAI23 Oral)

Learning to Sketch: A Neural Approach to Item Frequency Estimation in Streaming Data. (TPAMI24)

Mayfly: a Neural Data Structure for Graph Stream Summarization. (ICLR24 Spotlight)

## What is MetaSketch?
We pioneer the use of self-supervised memory neural networks for data stream compression, also known as sketching. Our results demonstrate that **this end-to-end learned compression paradigms** can offer greater potential than traditional **handcrafted methods**. In particular, MetaSketch achieves notably lower error rates under constrained memory budgets.
## Limitations
It is important to note that end-to-end learned data stream compression algorithms are still in their early stages and face certain limitations. For instance, the memory scalability required for real-world deployment may lead to frequent retraining. Nonetheless, we believe that ongoing research will continue to address these challenges and advance the end-to-end learning paradigm. For instance, our latest work [LegoSketch](https://github.com/FFY0/LegoSketch_ICML) (ICML25) effectively resolves the scalability issue and also further reduces compression error.

<img src="https://github.com/FFY0/LegoSketch_ICML/raw/main/Asserts/summarization.png" alt="Error comparision" width="50%">

## If you find this repo helpful, please kindly cite:

```

@inproceedings{cao2023meta,
  title={Meta-sketch: A neural data structure for estimating item frequencies of data streams},
  author={Cao, Yukun and Feng, Yuan and Xie, Xike},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={6},
  pages={6916--6924},
  year={2023}
}

@ARTICLE{10499867,
  author={Cao, Yukun and Feng, Yuan and Wang, Hairu and Xie, Xike and Zhou, S. Kevin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning to Sketch: A Neural Approach to Item Frequency Estimation in Streaming Data}, 
  year={2024},
  volume={46},
  number={11},
  pages={7136-7153},
  keywords={Streams;Vectors;Data structures;Artificial neural networks;Frequency estimation;Task analysis;Streaming media;Neural data structure;sketches;meta-learning;memory-augmented neural networks},
  doi={10.1109/TPAMI.2024.3388589}}


@inproceedings{feng2023mayfly,
  title={Mayfly: a neural data structure for graph stream summarization},
  author={Feng, Yuan and Cao, Yukun and Hairu, Wang and Xie, Xike and Zhou, S Kevin},
  booktitle={The twelfth international conference on learning representations},
  year={2023}
}

```
