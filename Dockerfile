
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN pip install torchtext==0.17.2 torchdata==0.7.1 --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install ipython ipykernel numpy==1.26.4 scipy pandas==2.3.1 scikit-learn tqdm==4.66.5 \
  transformers tokenizers datasets==2.19.1 sentencepiece==0.2.1 \
  langcodes==3.5.0 language-data==1.3.0 portalocker \
  matplotlib seaborn tensorboard==2.20.0 wandb spacy==3.7.4 

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download fr_core_news_sm
