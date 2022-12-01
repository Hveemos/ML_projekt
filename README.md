# ML_projekt
Curated scripts:
- read_data2.ipynb
- train_doc2vec.py
- train_sentenceKB_meta.py
- train_sentenceKB.py
- analyze_doc2vec.ipynb

# Get started

`git clone https://github.com/Hveemos/ML_projekt.git`

`pip install -r requirements.txt`

Get the data and assert correct folder structure by following the instructions in `read_data2.ipynb`.

Now you can start training and analyzing data with your models! For instance, you can run the `train_sentenceKB.py` script to train on all subparts from the documents. But beware! It might be a to large text vector. Consider slicing by rm first, e.g.: `df.loc[df.rm.isin(['2021/22','2020/21','2022','2021']),'dokdelar']`. Or you can simply try the `train_sentenceKB_meta.py` script but then you have to run the EXTRA part of `read_data2.ipynb`.
