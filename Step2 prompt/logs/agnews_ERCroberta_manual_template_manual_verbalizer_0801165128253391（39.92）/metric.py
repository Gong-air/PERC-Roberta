### put it into the predic directory with best checkpoint

from sklearn.metrics import f1_score, accuracy_score
preds = []
labels = []

'''
test_pred_epoch_batch size
'''

with open(file="test_preds.txt") as f:
    preds= f.read().splitlines()
with open(file="test_labels.txt") as f:
    labels= f.read().splitlines()
avg_accuracy = round(accuracy_score(labels,preds, )*100, 2)
avg_fscore = round(f1_score(labels, preds, average='weighted')*100, 2)
print("accuracy",avg_accuracy)
print("f1",avg_fscore)