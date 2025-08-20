# train_person_attr_mnv3s.py
# pip install torch torchvision tqdm
from pathlib import Path
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DS_ROOT = r"C:\Users\dalab\Desktop\Abbos\SmartLight\person_attr_fixed"
IMSIZE = 256
BATCH  = 64
EPOCHS = 12
LR     = 3e-4
OUT_WEIGHTS = r"C:\Users\dalab\Desktop\Abbos\SmartLight\person_attr_mnv3s.pth"

def get_loaders():
    tf_train = transforms.Compose([
        transforms.Resize((IMSIZE, IMSIZE)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    tf_val = transforms.Compose([
        transforms.Resize((IMSIZE, IMSIZE)),
        transforms.ToTensor()
    ])
    tr = datasets.ImageFolder(Path(DS_ROOT, "train"), transform=tf_train)
    va = datasets.ImageFolder(Path(DS_ROOT, "val"),   transform=tf_val)
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    va_ld = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    return tr_ld, va_ld, tr.classes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ld, va_ld, classes = get_loaders()
    print("Classes:", classes)  # ['no_device','uses_crutches','wheelchair_user']

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)

    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=LR)
    best_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        tr_loss = tr_corr = tr_tot = 0
        for x,y in tqdm(tr_ld, desc=f"Epoch {epoch} [train]"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out,y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*x.size(0)
            tr_corr += (out.argmax(1)==y).sum().item()
            tr_tot  += x.size(0)
        print(f"  train_acc={tr_corr/tr_tot:.3f}")

        # val
        model.eval()
        va_corr = va_tot = 0
        with torch.no_grad():
            for x,y in tqdm(va_ld, desc=f"Epoch {epoch} [val]"):
                x,y = x.to(device), y.to(device)
                out = model(x)
                va_corr += (out.argmax(1)==y).sum().item()
                va_tot  += x.size(0)
        va_acc = va_corr/va_tot
        print(f"  val_acc={va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": classes}, OUT_WEIGHTS)
            print(f"  -> saved {OUT_WEIGHTS} (best_acc={best_acc:.3f})")

if __name__ == "__main__":
    main()
