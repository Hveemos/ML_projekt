import pandas as pd
from pathlib import Path
import pandas as pd
from top2vec import Top2Vec
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Customize logger
#############################################################################################
logger = logging.getLogger('Train sentence-bert')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)
#############################################################################################

PATH_TO_DATA = Path('data')
INDATA = 'meta_prop_sou.pkl'

df=pd.read_pickle(PATH_TO_DATA / INDATA)
df=df.reset_index()

# Load model
sentence_model = SentenceTransformer('KBLab/sentence-bert-swedish-cased').encode
# Load tokenizer

tokenizer = AutoTokenizer.from_pretrained('KBLab/sentence-bert-swedish-cased').tokenize
model = Top2Vec(df['titel'].values, split_documents=False, embedding_model=sentence_model, use_embedding_model_tokenizer=True, tokenizer=tokenizer)

model_path=r"models\Modell_sbertKB_titel.mod"
model.save(model_path)

logger.info('Done!')
logger.info(f'Model saved to {model_path}')
logger.info(f'Metadata can be found in: {str(PATH_TO_DATA / INDATA)}')