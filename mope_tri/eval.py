from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from collections import Counter

import sys

def read_file(infile, ignore_GPE, remove_prefix):
    gold, pred1, pred2, pred3 = [], [], [], []
    with open(infile, "r") as inf:
        for line in inf:
            if not line.isspace() and not line.startswith('WORD\tGOLD') and not line.startswith('#sent-tok'):
                tok, g, p1, p2, p3 = line.strip().split("\t")
                if ignore_GPE and g.endswith('GPE'):
                    g = 'O'; p1 = 'O'; p2 = 'O'; p3 = 'O'
                if remove_prefix:
                    gold.append(g.strip("B-").strip("I-")); pred.append(p.strip("B-").strip("I-"))
                else:
                    gold.append(g); pred1.append(p1); pred2.append(p2); pred3.append(p3)
    return gold, pred1, pred2, pred3



def eval_seqeval(gold, pred, mode):
    f1 = f1_score([gold], [pred])
    if mode == 'strict':
        clf_report = classification_report([gold], [pred], mode='strict', digits=4, zero_division=0)
    else:
        clf_report = classification_report([gold], [pred], digits=4, zero_division=0)
    print(clf_report)


def get_majority_vote(p1, p2, p3):
    pred = []; prev_tag = 'O'
    for i in range(len(p1)):
        lst = [p1[i], p2[i], p3[i]]
        # check if we have a majority vote
        if len(Counter(lst)) == 3:
            pred.append(prev_tag)
        else:
            mf_tag = max(set(lst), key=lst.count)
            pred.append(mf_tag)
            prev_tag = mf_tag
    # sanity check
    if len(pred) != len(p1):
        print("LENGTH ERROR: ", len(pred), len(p1))
        sys.exit()
    return pred



### Main ###

result_file = sys.argv[1]
mode = "strict" # 'strict' for strict eval (punish incorrect prefixes)
ignore_GPE = False

gold, pred1, pred2, pred3 = read_file(result_file, ignore_GPE, False)
print("\nEVAL pred1:")
eval_seqeval(gold, pred1, mode)
print("\n\nEVAL pred2:")
eval_seqeval(gold, pred2, mode)
print("\n\nEVAL pred3:")
eval_seqeval(gold, pred3, mode)
pred = get_majority_vote(pred1, pred2, pred3)
print("\n\nEVAL majority vote:")
eval_seqeval(gold, pred, mode)



