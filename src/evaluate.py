y_true = all_labels
y_pred = all_preds
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculating the metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
uar = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Overall Precision: {precision:.4f}")
print(f"Unweighted Average Recall (UAR): {uar:.4f}")
print(f"Overall F1-Score: {f1:.4f}")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))
