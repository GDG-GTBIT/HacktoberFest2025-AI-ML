import nltk
nltk.download('punkt_tab')
import nltk
nltk.download('punkt_tab')
from transformers import BartForConditionalGeneration, BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
def bart_model(input_text):
    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=512, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output
import re
from nltk.tokenize import sent_tokenize
from summarygenerator import generate_summary
import rouge_score
def preprocess(text):
    text = remove_citations(text)
    text = split_long_sentences(text)
    text = replace_legal_terms(text)
    text = standardize_structure(text)
    return text

def postprocess(text):
    text = fix_formatting(text)
    text = ensure_consistency(text)
    text = add_paragraph_breaks(text)
    return text

def remove_citations(text):
    return re.sub(r'\(\d+\s+[A-Za-z\.]+\s+\d+\)', '', text)

def split_long_sentences(text):
    sentences = sent_tokenize(text)
    processed_sentences = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) > 50:
            splits = []
            current_split = []

            for token in sentence.split():
                current_split.append(token)
                if token == ',' or token == ';':
                    splits.append(' '.join(current_split))
                    current_split = []

            if current_split:
                splits.append(' '.join(current_split))
            processed_sentences.extend(splits)
        else:
            processed_sentences.append(sentence)

    return ' '.join(processed_sentences)

def replace_legal_terms(text):
    legal_terms = {
    'hereinafter': 'from now on',
    'pursuant to': 'according to',
    'whereas': 'since',
    'notwithstanding': 'despite',
    'forthwith': 'immediately',
    'inter alia': 'among other things',
    'ab initio': 'from the beginning',
    'ipso facto': 'by that fact itself',
    'mutatis mutandis': 'with the necessary changes',
    'de facto': 'in fact',
    'de jure': 'by law',
    'quid pro quo': 'something for something',
    'sub judice': 'under judicial consideration',
    'prima facie': 'at first glance',
    'pro rata': 'in proportion',
    'ultra vires': 'beyond the powers',
    'res judicata': 'a matter already judged',
    'a fortiori': 'even more so',
    'ex parte': 'by one party',
    'actus reus': 'guilty act',
    'mens rea': 'guilty mind',
    'nolo contendere': 'no contest',
    'stare decisis': 'to stand by decided cases',
    'in loco parentis': 'in the place of a parent',
    'per curiam': 'by the court',
    'amicus curiae': 'friend of the court',
    'sui generis': 'unique',
    'caveat emptor': 'let the buyer beware',
    'habeas corpus': 'you shall have the body',
    'ex post facto': 'after the fact',
    'in situ': 'in its original place',
    'pari passu': 'on equal footing',
    'lex loci': 'law of the place',
    'contra proferentem': 'against the drafter',
    'pro bono': 'for the public good',
    'ad hoc': 'for this specific purpose',
    'ex officio': 'by virtue of office',
    'jus cogens': 'compelling law',
    'locus standi': 'right to bring action',
    'nullum crimen sine lege': 'no crime without law',
}


    for term, replacement in legal_terms.items():
        text = re.sub(r'\b' + term + r'\b', replacement, text, flags=re.IGNORECASE)
    return text

def standardize_structure(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Section \d+\.', lambda m: '\n' + m.group(0) + '\n', text)
    return text

def fix_formatting(text):
    text = '. '.join(s.strip().capitalize() for s in text.split('. '))
    text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1 \2', text)
    return text

def ensure_consistency(text):
    return text

def add_paragraph_breaks(text):
    text = re.sub(r'([.!?])\s+(?=[A-Z])', r'\1\n\n', text)
    return text

def process_document(text, model):
    simplified_text = preprocess(text)
    model_output = bart_model(simplified_text)
    final_text = postprocess(model_output)
    return final_text
text = '''WHEREAS, the parties hereto agree to the terms and conditions set forth in this Agreement; and, pursuant to Section 12.3,
 all disputes arising hereunder shall be resolved through arbitration. NOTWITHSTANDING any provision to the contrary, the obligations
  herein shall commence forthwith. HEREINAFTER, the terms shall be interpreted according to the laws of the State of California.
   Section 14. This document also includes, inter alia, provisions for confidentiality and data protection.'''
res=process_document(text,model)
presimplification=generate_summary(res)
summary=generate_summary(text)
postsimplification=process_document(summary,model)

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores_pre = []
rouge_scores_post=[]
for gen, ref in zip(presimplification, text):
    score = scorer.score(gen, ref)
    rouge_scores_pre.append(score)
for gen, ref in zip(postsimplification, text):
    score = scorer.score(gen, ref)
    rouge_scores_post.append(score)
import numpy as np
avg_rouge_scores_pre = {
    'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores_pre]),
    'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores_pre]),
    'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores_pre])
}
print("\nAverage ROUGE Scores:")
for metric, score in avg_rouge_scores_pre.items():
    print(f"{metric}: {score:.4f}")
avg_rouge_scores_post = {
    'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores_post]),
    'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores_post]),
    'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores_post])
}
print("\nAverage ROUGE Scores:")
for metric, score in avg_rouge_scores_post.items():
    print(f"{metric}: {score:.4f}")
summary_results_pre= {
    'generated_summaries': presimplification,
    'reference_summaries': text,
    'rouge_scores': avg_rouge_scores_pre
}
summary_results_post = {
    'generated_summaries': postsimplification,
    'reference_summaries': text,
    'rouge_scores': avg_rouge_scores_post
}
summary_results_pre
summary_results_post