# Chinese gigaword corpus processing
We processed and filtered the [Chinese Gigaword Third Edition](https://catalog.ldc.upenn.edu/LDC2007T38) and finally yield 227,000 pairs of (S, H).

Requirements:

```bash
pip install stanza==1.1.1
pip install jieba==0.42.1
```

Also, download [dict.txt.big](https://github.com/fxsjy/jieba/blob/master/extra_dict/dict.txt.big) and put it in your OWN_DIRECTARY.

Then, put data_processing.py and gigaword raw data zbn_cmn, cna_cmn, afp_cmn, and xin_cmn in the same folder and run

```bash
python data_processing.py
```

As a result of data processing, we yielded two files:
- data_s_parsed.txt, 227,000 sentences
- data_h_parsed.txt, 227,000 headlines

