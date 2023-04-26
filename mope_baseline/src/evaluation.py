
def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}



"""
Gold ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-EPPOL', 'I-EPPOL', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-EPPOL', 'I-EPPOL', 'I-EPPOL', 'O', 'O', 'O', 'O', 'B-EPPOL', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'O', 'O', 'O', 'B-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'I-PFUNK', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
Auto ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
RES  {'tp': 0, 'fp': 0, 'fn': 0}
"""

def mope_evaluation(gold_original, system_original, result_dic):
    print("Gold", gold_original)
    print("Auto", system_original)
    print("RES ", result_dic)
    auto_list = []
    
    gold_list = []
    
    tp = 0
    fp = 0
    fn = 0

    auto = system_original
    gold = gold_original
    
    i = 0
    while i < len(gold_original):
        gold[i] = gold_original[i].strip("B-").strip("I-")
        #gold[i] = gold_original[i].strip("I-")
        gold_list.append(i)
        i = i+1

    j = 0
    while j < len(system_original):
        auto[j] = system_original[j].strip("B-").strip("I-")
        #auto[j] = system_original[j].strip("I-")
        auto_list.append(j)

        j = j+1
    
    auto_list.sort()
    gold_list.sort()
    
    print("Gold source: ", gold_list)
    print("Gold target: ", gold_list)
    print("System source: ", auto_list)
    print("System target: ", auto_list)
    print("\n")
    
    tp, fp, fn = 0, 0, 0

    
    result_dic["tp"] += tp
    result_dic["fp"] += fp
    result_dic["fn"] += fn
    
    return tp, fp, fn, result_dic


# Calculates the micro average of recall, precision and F1 for source and target predictions

def micro_average(result_dic):
    tp = result_dic["tp"]
    fp = result_dic["fp"]
    fn = result_dic["fn"]
    recall = 0
    precision = 0
    f1 = 0
    
    if tp > 0: 
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        if (recall + precision) > 0:
          f1 = (2*recall*precision)/(recall+precision)
    
    
    print("MOPE Results:")
    print("TP: {}, FP: {}, FN: {}".format(tp, fp, fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(recall, 3), round(precision, 3), round(f1, 3)))
    print("")

    return recall, precision, f1

# Evaluation on token level
# No distinction between the individual roles

def srl_micro_average(dict_srl):
    srl_tp = dict_srl["srl_tp"]
    srl_fp = dict_srl["srl_fp"]
    srl_fn = dict_srl["srl_fn"]
    srl_recall = 0
    srl_precision = 0
    srl_f1 = 0

    if srl_tp > 0:
        srl_recall = srl_tp/(srl_tp+srl_fn)
        srl_precision = srl_tp/(srl_tp+srl_fp)
        if (srl_recall + srl_precision) > 0:
          srl_f1 = (2*srl_recall*srl_precision)/(srl_recall+srl_precision)

    print("SRL Results:")
    print("TP: {}, FP: {}, FN: {}".format(srl_tp, srl_fp, srl_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(srl_recall, 3), round(srl_precision, 3), round(srl_f1, 3)))
    print("\n")

    return srl_recall, srl_precision, srl_f1



def dep_evaluation(gold_original, system_original, dict_dep):
    dep_tp = 0
    dep_fp = 0
    dep_fn = 0
    system = system_original
    gold = gold_original

    j = 0
    while j < len(system_original):
      system[j] = system_original[j].strip("B-")
      system[j] = system_original[j].strip("I-")
      j = j+1


    i = 0
    while i < len(gold_original):
        gold[i] = gold_original[i].strip("B-")
        gold[i] = gold_original[i].strip("I-")


        if i < len(system):
          if gold[i] != "O" and gold[i] != "X":
            if gold[i] == system[i]:
              dep_tp = dep_tp + 1
            elif system[i] == "O":
                dep_fn = dep_fn +1
            else:
                dep_fn = dep_fn +1
                dep_fp = dep_fp +1

          else:
            if gold[i] == system[i]: # both "O" or "X"
              pass
            else:
             dep_fp = dep_fp + 1

        i = i+1

    dict_dep["dep_tp"] = dict_dep["dep_tp"] + dep_tp
    dict_dep["dep_fp"] = dict_dep["dep_fp"] + dep_fp
    dict_dep["dep_fn"] = dict_dep["dep_fn"] + dep_fn

    return dep_tp, dep_fp, dep_fn, dict_dep



def dep_micro_average(dict_dep):
    dep_tp = dict_dep["dep_tp"]
    dep_fp = dict_dep["dep_fp"]
    dep_fn = dict_dep["dep_fn"]
    dep_recall = 0
    dep_precision = 0
    dep_f1 = 0

    if dep_tp > 0:
        dep_recall = dep_tp/(dep_tp+dep_fn)
        dep_precision = dep_tp/(dep_tp+dep_fp)
        if (dep_recall + dep_precision) > 0:
          dep_f1 = (2*dep_recall*dep_precision)/(dep_recall+dep_precision)

    print("DEP Results:")
    print("TP: {}, FP: {}, FN: {}".format(dep_tp, dep_fp, dep_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(dep_recall, 3), round(dep_precision, 3), round(dep_f1, 3)))
    print("\n")

    return dep_recall, dep_precision, dep_f1





# Evaluation on token level
# No distinction between the individual roles

def srl_evaluation(gold_original, system_original, dict_srl):

    srl_tp = 0
    srl_fp = 0
    srl_fn = 0
    system = system_original
    gold = gold_original
    
    j = 0
    while j < len(system_original):
      system[j] = system_original[j].strip("B-")
      system[j] = system_original[j].strip("I-")
      j = j+1


    i = 0
    while i < len(gold_original):
        gold[i] = gold_original[i].strip("B-")
        gold[i] = gold_original[i].strip("I-")


        if i < len(system):
          if gold[i] != "O" and gold[i] != "X":
            if gold[i] == system[i]:
              srl_tp = srl_tp + 1
            elif system[i] == "O":
                srl_fn = srl_fn +1
            else:
                srl_fn = srl_fn +1
                srl_fp = srl_fp +1
                
          else:
            if gold[i] == system[i]: # both "O" or "X"
              pass
            else:
             srl_fp = srl_fp + 1
       
        i = i+1
        
    dict_srl["srl_tp"] = dict_srl["srl_tp"] + srl_tp
    dict_srl["srl_fp"] = dict_srl["srl_fp"] + srl_fp
    dict_srl["srl_fn"] = dict_srl["srl_fn"] + srl_fn
                
    return srl_tp, srl_fp, srl_fn, dict_srl



def srl_micro_average(dict_srl):
    srl_tp = dict_srl["srl_tp"]
    srl_fp = dict_srl["srl_fp"]
    srl_fn = dict_srl["srl_fn"]
    srl_recall = 0
    srl_precision = 0
    srl_f1 = 0
    
    if srl_tp > 0: 
        srl_recall = srl_tp/(srl_tp+srl_fn)
        srl_precision = srl_tp/(srl_tp+srl_fp)
        if (srl_recall + srl_precision) > 0:
          srl_f1 = (2*srl_recall*srl_precision)/(srl_recall+srl_precision)
    
    print("SRL Results:")
    print("TP: {}, FP: {}, FN: {}".format(srl_tp, srl_fp, srl_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(srl_recall, 3), round(srl_precision, 3), round(srl_f1, 3)))
    print("\n")

    return srl_recall, srl_precision, srl_f1



