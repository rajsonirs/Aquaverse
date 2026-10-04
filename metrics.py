"""Basic labeled-reading evaluation metrics."""
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def evaluate(results):
    if "leak_label" not in results or results.leak_label.isna().all(): return None
    y=results.leak_label.fillna(0).astype(int); p=results.alert.astype(int)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"precision":round(precision_score(y,p,zero_division=0),3),"recall":round(recall_score(y,p,zero_division=0),3),"f1":round(f1_score(y,p,zero_division=0),3),"false_alarms":int(fp),"missed_leak_readings":int(fn),"detected_leak_readings":int(tp),"normal_readings_correct":int(tn)}
