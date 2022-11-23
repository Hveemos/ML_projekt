import pandas as pd
from pathlib import Path
import pandas as pd
from gensim.utils import simple_preprocess
from functools import partial
from top2vec import Top2Vec
import logging

# Customize logger
#############################################################################################
logger = logging.getLogger('Train Doc2Vec')
logger.setLevel(logging.WARNING)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)
#############################################################################################

PATH_TO_OUTDATA=Path('data')

df=pd.read_pickle(PATH_TO_OUTDATA / 'data.pkl')
df=df.reset_index()


# By default Top2Vec uses gensim.utils.simple_preprocess to tokenize.
# But in Top2Vec deacc is set to True
# We use functools to change the default setting
tok=partial(simple_preprocess,deacc=False) # Change default value and instantiate new function tok

model = Top2Vec(df['text'].values, split_documents=True, tokenizer=tok)

model_path=r"models\Modell_deaccFalse_11_22_doc2vec.mod"
model.save(model_path)

logger.info('Done!')
logger.info('Model saved to', model_path)
logger.info('Metadata can be found in:', str(PATH_TO_OUTDATA / 'data.pkl'))