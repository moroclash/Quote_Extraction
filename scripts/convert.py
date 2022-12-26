from pathlib import Path
from typing import List

import typer
import pandas as pd
import re
import os
from spacy.tokens import Doc, DocBin, Span
import spacy
from sklearn.model_selection import train_test_split
from collections import Counter
from wasabi import msg
import warnings
warnings.filterwarnings('ignore')


NAMES_MAPPER = {"Cue": "CUE",
                "Quotation" : "QUOTE",
                "Entity": "SPEAKER",
                "Out": "OUT",
                "LeftSpeaker": "QUOTE",
                "RightSpeaker": "QUOTE",
                "Unknown": "QUOTE",
                "Speaker": "SPEAKER",
                "B-source": "B-SPEAKER",
                "I-source": "I-SPEAKER",
                "B-cue": "B-CUE",
                "I-cue": "I-CUE",
                "B-content": "B-QUOTE",
                "I-content": "I-QUOTE",
                }


def convert_DirectQuote_to_docs(
    data: str,
    Cues_verbs: list = []
) -> List[Doc]:
    """Parse DirectQuote dataset into spaCy docs
    """
    
    def clean_data(lines):
        docs = []
        doc_id = 1
        for line in lines:
            if line == '\n':
                doc_id +=1
            else:
                docs.append(line.replace('\n', '').split(' ') + [doc_id])
        return docs

    def convert_data_to_spacy_format_nets(data_):
        nlp = spacy.blank("en") # load a new spacy model
        docs = [] # create a DocBin object

        for doc_id, df_group in data_.groupby('doc_id'):
            spaces = [False if i in ['.', ','] else True for i in df_group.word]
            ents = ['O' if t == 'Out' else t for t in df_group.tag]
            doc = Doc(nlp.vocab, words=df_group.word, spaces=spaces, ents=ents)        
            docs.append(doc)
        return docs, nlp


    def convert_data_to_spacy_formate_spans(data_):
        nlp = spacy.blank("en") # load a new spacy model
        # nlp = spacy.load("en_core_web_sm") # load other spacy model
        docs = [] # create a DocBin object

        for doc_id, df_group in data_.groupby('doc_id'):
            doc = nlp.make_doc(' '.join(df_group.word)) # create doc object from text
            spans = []

            # loop to get entities
            start, end, last_tag = 0, 0, None
            for i, row in df_group.reset_index().iterrows():
                tag = NAMES_MAPPER[row.tag]
                if tag != 'OUT':
                    if last_tag == None and i+1 != len(df_group):
                        last_tag = tag
                        start = i
                        end = i

                    elif tag == last_tag:
                        end = i
                    else:
                        if i+1 == len(df_group):
                            start, end, last_tag = i, i, tag
                        spans.append(Span(doc, start, end+1, label=last_tag))
                        start, end, last_tag = i, i, tag
                elif last_tag != None:
                    spans.append(Span(doc, start, end+1, label=last_tag))
                    start, end, last_tag = start, end, None
            
            doc.spans["sc"] = spans
            docs.append(doc)
        return docs, nlp


    docs = clean_data(data)
    data = pd.DataFrame(data=docs, columns=['word', 'tag', 'doc_id'])
    # remove non important chars
    data.tag = data.tag.map(lambda x: re.sub(r'.-', '', x))

    if len(Cues_verbs) > 0:
        # adding verb Cue to dataset
        for doc_id, df_group in data.groupby('doc_id'):
            if len(set(df_group.tag)) > 1:
                data.tag.iloc[df_group[df_group.word.map(lambda x: x.lower() in Cues_verbs) == True].index] = 'Cue'

    docs, nlp = convert_data_to_spacy_formate_spans(data)
    return docs



# def convert_RiQuA_to_docs(
#     folder_path: str,
# ) -> List[Doc]:
#     """Parse RiQuA dataset into spaCy docs
#     """

#     nlp = spacy.blank("en")
#     all_docs = []
#     all_entities = []

#     for file_ in filter(lambda x: x[-3:] == 'txt' ,os.listdir(folder_path)):
#         # read content of each file
#         with open(f'{folder_path}/{file_}', 'r') as f:
#             content = f.readlines()[0]
    
#         doc = nlp.make_doc(content)

#         with open(f'{folder_path}/{file_[:-4]}.ann', 'r') as f:
#             annotations = f.readlines()
    
#         entities = list(map(lambda x: x.split('\t')[1].split(' '), filter(lambda x: x[0] == 'T' , annotations)))

#         all_entities += list(filter(lambda x: x[0] == 'T' , annotations))
    
#         spans = []
#         for tag, start, end in entities:
#             tag = NAMES_MAPPER[tag]
#             span = doc.char_span(int(start), int(end), label=tag, alignment_mode='expand')
#             if span != None:
#                 spans.append(Span(doc, span.start, span.end, label=tag))
#         doc.spans["sc"] = spans
#         all_docs.append(doc)

#     Cues_verbs = set(i[0].lower() for i in filter(lambda x: x[1] == 'Cue', map(lambda x: (x.split('\t')[2].replace('\n', ''), x.split('\t')[1].split()[0]), all_entities)))

#     return all_docs, Cues_verbs

def convert_RiQuA_to_docs(
    folder_path: str,
) -> List[Doc]:
    """Parse RiQuA dataset into spaCy docs
    """

    with open(folder_path, 'r') as f:
        data_ = f.readlines()
    
    nlp = spacy.blank("en")
    docs = []
    Cues_verbs = []
    for row in data_:
        row = eval(row)
        tokens, tags = row['tokens'], row['labels']
        tags = [NAMES_MAPPER[t] if t in NAMES_MAPPER else t for t in tags]
        spaces = [True]*len(tokens)
        doc = Doc(nlp.vocab, words=tokens, spaces=spaces, ents=tags)
        spans = [Span(doc, ent.start, ent.end, label=ent.label_) for ent in doc.ents]
        doc.spans["sc"] = spans
        docs.append(doc)
        Cues_verbs += map(lambda x: x.text.lower(), filter(lambda x: x.label_ == 'CUE' ,doc.ents))
    return docs, Cues_verbs




def convert_Polnear_to_docs(
    folder_path: str,
) -> List[Doc]:
    """Parse Polnear dataset into spaCy docs
    """

    with open(file_path, 'r') as f:
        data_ = f.readlines()
    
    nlp = spacy.blank("en")
    docs = []
    Cues_verbs = []
    for row in data_:
        row = eval(row)
        tokens, tags = row['tokens'], row['labels']
        tags = [NAMES_MAPPER[t] if t in NAMES_MAPPER else t for t in tags]
        spaces = [True]*len(tokens)
        doc = Doc(nlp.vocab, words=tokens, spaces=spaces, ents=tags)
        spans = [Span(doc, ent.start, ent.end, label=ent.label_) for ent in doc.ents]
        doc.spans["sc"] = spans
        docs.append(doc)
        Cues_verbs += map(lambda x: x.text.lower(), filter(lambda x: x.label_ == 'CUE' ,doc.ents))
    return docs, Cues_verbs


def Train_Test_Split(all_docs:List[Doc], split_ratio:float=.3):
    #Split the data into the training set (90%) and validation set (10%)
    train_set, test_set = train_test_split(all_docs, test_size=split_ratio)
    # #Split the validation set into the actual validation set (70%) and test set (30%)
    # validation_set, test_set = train_test_split(validation_set, test_size=0.3)
    #Print how many docs are in each set
    print(f'🚂 Created {len(train_set)} training docs')
    print("Train set counts")
    print(Counter([s.label_ for x in train_set for s in x.spans['sc']]))
    # print(f'😊 Created {len(validation_set)} validation docs')
    print(f'🧪 Created {len(test_set)} test docs')
    print("Test set counts")
    print(Counter([s.label_ for x in test_set for s in x.spans['sc']]))
    return train_set, test_set


def main(
    test_ratio: float = typer.Option(..., "--split-ratio", exists=True),
    output_path: Path = typer.Option(..., "-o", "--output", exists=True),
):
    all_Cue_verbs = []
    all_docs = []

    with msg.loading(f"Processing RiQuA Dataset..."):
        len_docs_RiQuA = 0
        docs_RiQuA, Cue_verbs = convert_RiQuA_to_docs('assets/riqua_train.txt')
        len_docs_RiQuA += len(docs_RiQuA)
        all_docs += docs_RiQuA
        all_Cue_verbs += Cue_verbs 

        docs_RiQuA, Cue_verbs = convert_RiQuA_to_docs('assets/riqua_test.txt')
        len_docs_RiQuA += len(docs_RiQuA)
        all_docs += docs_RiQuA
        all_Cue_verbs += Cue_verbs 

        docs_RiQuA, Cue_verbs = convert_RiQuA_to_docs('assets/riqua_valid.txt')
        len_docs_RiQuA += len(docs_RiQuA)
        all_docs += docs_RiQuA
        all_Cue_verbs += Cue_verbs 
    msg.good(f"Processing Done :: {len_docs_RiQuA} Docs")


    with msg.loading(f"Processing Polnear Dataset..."):
        len_docs_polnear = 0
        docs_polnear, Cue_verbs = convert_Polnear_to_docs('assets/polnear_train.txt')
        len_docs_Polnear += len(docs_polnear)
        all_docs += docs_polnear
        all_Cue_verbs += Cue_verbs 

        docs_polnear, Cue_verbs = convert_Polnear_to_docs('assets/polnear_test.txt')
        len_docs_Polnear += len(docs_polnear)
        all_docs += docs_polnear
        all_Cue_verbs += Cue_verbs 

        docs_polnear, Cue_verbs = convert_Polnear_to_docs('assets/polnear_valid.txt')
        len_docs_Polnear += len(docs_polnear)
        all_docs += docs_polnear
        all_Cue_verbs += Cue_verbs 
    msg.good(f"Processing Done :: {len_docs_polnear} Docs")


    with msg.loading(f"Processing DirectQuote Dataset..."):
        with open('assets/DirectQuote/truecased.txt', "r", encoding="utf-8") as f:
            data = f.readlines()
        docs_DirectQuote = convert_DirectQuote_to_docs(data, set(all_Cue_verbs))
        all_docs += docs_DirectQuote
    msg.good(f"Processing Done :: {len(docs_DirectQuote)} Docs")

    with msg.loading(f"Train Test Split for :: {len(all_docs)} Docs  with test Ration {test_ratio}...."):
        train_set, test_set = Train_Test_Split(all_docs, float(test_ratio))
    msg.good(f"Spliting Done")

    with msg.loading(f"Saving into DocBin..."):
        doc_bin = DocBin(docs=train_set)
        doc_bin.to_disk(f"{output_path}/train.spacy")
        msg.good(f"Saved to {output_path}/train.spacy")

        doc_bin = DocBin(docs=test_set)
        doc_bin.to_disk(f"{output_path}/test.spacy")
        msg.good(f"Saved to {output_path}/test.spacy")


if __name__ == "__main__":
    typer.run(main)