import json
import jsonlines
import numpy as np

corpus = {doc['doc_id']: doc for doc in jsonlines.open("./corpus.jsonl")}
sci_dev = jsonlines.open("./scifact_dev.jsonl")

with open("./dev_cogat.json", "w") as fout:
    for line in sci_dev:
        evidence_list = []
        # sci = json.loads(line)
        id = line["id"]
        # print(id)
        claim = line["claim"]
        label = line["label"]
        evidence = line['evidence']
        for evi in evidence:
            did = evi[0]
            title = evi[1]
            sid = evi[2]
            e = evi[3]
            dataset = jsonlines.open("./claims_dev.jsonl")
            for data in dataset:
                if id == data['id']:
                    if len(data["evidence"].keys()) == 0:
                        evidence_list.append([did,title, sid, e, 0])
                    else:
                        if str(did) in data["evidence"].keys():
                            flag = False
                            for sentence in data["evidence"][str(did)]:
                                if sid in sentence["sentences"]:
                                    flag = True
                                    evidence_list.append([did,title, sid, e, 1.0])
                                    break
                            if flag == False:
                                evidence_list.append([did,title, sid, e, 0])
                        else:
                            evidence_list.append([did,title, sid, e, 0])
        fout.write(json.dumps(
            {"id": id, "claim": claim, "evidence": evidence_list, "label": label}) + "\n")


sci_train = jsonlines.open("./scifact_train.jsonl")

with open("cogat_train.json", "w") as fout:
    for line in sci_train:
        evidence_list = []
        # sci = json.loads(line)
        id = line["id"]
        # print(id)
        claim = line["claim"]
        label = line["label"]
        evidence = line['evidence']
        for evi in evidence:
            did = evi[0]
            title = evi[1]
            sid = evi[2]
            e = evi[3]
            dataset = jsonlines.open("../../data/claims_train.jsonl")
            for data in dataset:
                if id == data['id']:
                    if len(data["evidence"].keys()) == 0:
                        evidence_list.append([did,title, sid, e, 0])
                    else:
                        if str(did) in data["evidence"].keys():
                            flag = False
                            for sentence in data["evidence"][str(did)]:
                                if sid in sentence["sentences"]:
                                    flag = True
                                    evidence_list.append([did,title, sid, e, 1.0])
                                    break
                            if flag == False:
                                evidence_list.append([did,title, sid, e, 0])
                        else:
                            evidence_list.append([did,title, sid, e, 0])
        fout.write(json.dumps(
            {"id": id, "claim": claim, "evidence": evidence_list, "label": label}) + "\n")

scifact_data =list()
fever_data = list()
with open("./cogat_train.json") as fin:
    scifact_data = fin.readlines()
with open("./fever.jsonl") as fin:
    fever_data = fin.readlines()

scifact_data = scifact_data * 50
fever_data = fever_data + scifact_data
np.random.shuffle(fever_data)
with open("./train_cogat.json", "w") as fout:
    for step,data in enumerate(fever_data):
        print(step)
        fout.write(data)
