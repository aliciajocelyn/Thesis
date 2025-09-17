import rbo
import itertools
from itertools import combinations
import numpy as np

from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from gensim import corpora

def calculate_coherence_score(texts, all_topics, coherence_types=None, print_results=False):
  coherence_types = ['c_v', 'u_mass', 'c_uci', 'c_npmi']
  
  tokenized_docs = [simple_preprocess(doc) for doc in texts]

  # Build dictionary and corpus
  dictionary = corpora.Dictionary(tokenized_docs)
  corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

  scores = {}

  # Compute all coherence types
  for coherence in coherence_types:
      model = CoherenceModel(
          topics=all_topics,
          texts=tokenized_docs,
          corpus=corpus,
          dictionary=dictionary,
          coherence=coherence
      )
      scores[coherence] = round(model.get_coherence(), 3)

  if print_results:
    for metric, score in scores.items():
      print(f"{metric}: {score}")
      
  return scores

def calculate_irbo(all_topics, topk=10, print_results=False):
    """
    all_topics: list of topic words (each topic is a list of top-k words)
    topk: how many top words to use per topic
    """
    T = len(all_topics)
    n = T * (T - 1) / 2
    rbo_sum = 0

    for i in range(1, T):
        for j in range(i):
            l1 = all_topics[i][:topk]
            l2 = all_topics[j][:topk]
            rbo_score = rbo.RankingSimilarity(l1, l2).rbo()
            rbo_sum += rbo_score

    irbo_score = 1 - (rbo_sum / n)
    irbo_score = round(irbo_score, 3)

    if print_results: 
      print(f"IRBO: {irbo_score}")
      
    return irbo_score

def evaluate_topics(texts, topic_model, coherence_types=None, topk=10):
    coherence_types=['c_v', 'u_mass', 'c_uci', 'c_npmi']
  
    # Tokenized text for coherence
    tokenized_docs = [simple_preprocess(doc) for doc in texts]
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    # Get all topic words
    all_topics = [ [word for word, _ in topic_model.get_topic(topic_id)[:topk]]
                   for topic_id in topic_model.get_topics().keys()
                   if topic_id != -1 and topic_model.get_topic(topic_id) is not None ]

    # Topic Coherence
    coherence_scores = {}
    for ctype in coherence_types:
        cm = CoherenceModel(topics=all_topics, texts=tokenized_docs, corpus=corpus, dictionary=dictionary, coherence=ctype)
        coherence_scores[ctype] = round(cm.get_coherence(), 3)

    # Topic Diversity (IRBO)
    irbo_score = calculate_irbo(all_topics, topk=topk)

    # Print all
    for ctype, score in coherence_scores.items():
        print(f"Coherence ({ctype}): {score}")
    print(f"IRBO Topic Diversity: {irbo_score}")

    return coherence_scores, irbo_score
