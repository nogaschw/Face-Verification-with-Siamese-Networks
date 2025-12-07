from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix

def train(model, criterion, train_loader, val_loader):   

    def call_model(img1, img2, label):
        img1, img2, label = img1.to(device), img2.to(device), label.float().to(device)
        output = model(img1, img2).squeeze()
        loss = criterion(output, label)
        preds = (output.detach().cpu().numpy() >= 0.5).astype(int)
        labels = label.cpu().numpy()
        return loss, preds, labels

    best_loss = float('inf')
    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        print(f'number of epoch: {epoch}')
        model.train()

        train_loss = 0.0
        all_preds = []
        all_labels = []
        for img1, img2, label in tqdm(train_loader):
            optimizer.zero_grad()
            loss, preds, labels = call_model(img1, img2, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_train_loss = train_loss / len(train_loader)
        accuracy_train.append(accuracy_score(all_labels, all_preds))

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                loss, preds, labels = call_model(img1, img2, label)
                val_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels)

        accuracy_val.append(accuracy_score(all_labels, all_preds))
        avg_val_loss = val_loss / len(val_loader)
        l_train_loss.append(avg_train_loss)
        l_val_loss.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train Accuracy={accuracy_train[-1]:.4f}, Val Accuracy={accuracy_val[-1]:.4f}")

        # Early stopping check
        # if avg_val_loss < best_loss:
        #     best_loss = avg_val_loss
        # Early stopping base accuracy
        if accuracy_val[-1] > best_acc:
            best_acc = accuracy_val[-1]
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        print(f"Epoch {epoch+1}: lr = {optimizer.param_groups[0]['lr']:.6f}")

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            outputs = model(img1, img2).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert predictions to binary (threshold 0.5)
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]

    accuracy = accuracy_score(all_labels, bin_preds)
    cm = confusion_matrix(all_labels, bin_preds)

    print("Confusion Matrix:")
    print(cm)

    return {
        'Accuracy': accuracy,
        'ConfusionMatrix': cm
    }
