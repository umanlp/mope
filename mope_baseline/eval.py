from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import sys

def read_file(infile, ignore_GPE):
    gold, pred = [], []
    with open(infile, "r") as inf:
        for line in inf:
            if not line.isspace() and not line.startswith('WORD\tGOLD') and not line.startswith('#sent-tok'):
                tok, g, p = line.strip().split("\t")
                if ignore_GPE and g.endswith('GPE'):
                    g = 'O'; p1 = 'O';
                gold.append(g); pred.append(p)
    return gold, pred


def eval_seqeval(gold, pred, mode):
    f1 = f1_score([gold], [pred])
    if mode == 'strict':
        print("STRICT EVALUATION")
        clf_report = classification_report([gold], [pred], mode='strict', digits=4)
    else:
        print("NON-STRICT (ignore prefixes)")
        clf_report = classification_report([gold], [pred], digits=4)
    print(clf_report)



##### Main #####

result_file = sys.argv[1]
mode = "strict"     # 'strict' for strict eval (punish incorrect prefixes)
ignore_GPE = False  # set to 'True' to ignore GPE tags in the evaluation

gold, pred = read_file(result_file, ignore_GPE)
eval_seqeval(gold, pred, mode)


