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
N_TOPICS=32 #Number of topics

df=pd.read_pickle(PATH_TO_DATA / INDATA)
df=df.set_index('dok_id') # Set correct index

logger.info(f'Loaded {df.shape[0]} documents')


# Load model
sentence_model = SentenceTransformer('KBLab/sentence-bert-swedish-cased').encode
# Load tokenizer

tokenizer = AutoTokenizer.from_pretrained('KBLab/sentence-bert-swedish-cased').tokenize
model = Top2Vec(df['titel'].values, document_ids=df.index.to_list(), split_documents=False, embedding_model=sentence_model, use_embedding_model_tokenizer=True, tokenizer=tokenizer, workers=8)

model_path=r"models\Modell_sbertKB_titel.mod"


# Reduce number of topics to N_TOPICS
logger.info('Reduce number of topics')
model.hierarchical_topic_reduction(N_TOPICS)

model.save(model_path)

logger.info('Done!')
logger.info(f'Model saved to {model_path}')
logger.info(f'Metadata can be found in: {str(PATH_TO_DATA / INDATA)}')