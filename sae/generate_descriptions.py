# %%
import torch as t
import numpy as np
import argparse
from openai import OpenAI
from tqdm import tqdm
import json
import pandas as pd
import openai
import heapq
import gzip


def main():
    parser = argparse.ArgumentParser(description="Generate latent descriptions")

    parser.add_argument(
        "--passages",
        type=str,
        required=True,
        help="msmarco raw passages filepath",
    )
    parser.add_argument(
        "--latent_concepts",
        type=str,
        required=True,
        help="file path with each document's activated latent",
    )
    parser.add_argument(
        "--topK",
        type=int,
        required=True,
        help="show up to top K documents to generate explanation from",
    )
    parser.add_argument(
        "--api",
        type=str,
        required=True,
        help="OpenAI api key",
    )

    client = OpenAI(api_key=args.api)

    raw_passages = []
    with gzip.open(args.latent_concepts, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            raw_passages.append(data)

    interp_list = []

    tempdict = defaultdict(list)

    with open(args.latent_concepts, "r") as file:
        for line in tqdm(file):
            entry = json.loads(line)
            for i, latent_id in enumerate(entry["ids"]):
                tempdict[latent_id].append((entry["docid"], entry["weight"][i]))

    for id, lst in tqdm(tempdict.items()):
        entry = {"id": id, "doc_weights": lst}
        latent_id = entry["id"]
        docs = entry["doc_weights"]
        topK = heapq.nlargest(args.topK, docs, key=lambda x: x[1])
        doc_texts = []
        doc_acts = []
        for docid, act in topK:
            doc_texts.append(raw_passages[docid]["contents"])
            doc_acts.append(act)
        request_text = ""
        for i, (text, act) in enumerate(zip(doc_texts, doc_acts)):
            if i >= args.topK:
                break
            request_text += f"Example{i}: {text}\nActivation: {act}\n"

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "developer",
                    "content": """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an interpretation that thoroughly encapsulates possible patterns found in it.
Guidelines:
You will be given a list of text examples on which a certain common pattern might be present. How important each text is for the pattern is listed after each text.
- Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don’t need to mention them. Don’t focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Based on the found patterns summarize your interpretation in 1-8 words.
- Do not make lists of possible interpretations. Keep your interpretations short and concise.
- The last line of your response must be the formatted interpretation, using [interpretation]:
""",
                },
                {
                    "role": "user",
                    "content": request_text,
                },
            ],
        )
        # print(completion)
        interp_text = completion.choices[0].message.content
        interp_list.append((latent_id, interp_text))

    with open(
        args.out,
        "w",
    ) as f:
        for id, text in interp_list:
            only_interp = text.split("[interpretation]:")[-1].strip()
            # json.dump({"id": id, "response": text, "interp": only_interp}, f)
            json.dump({"id": id, "interp": only_interp}, f)
            f.write("\n")
