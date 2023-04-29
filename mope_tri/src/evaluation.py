
def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}


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



