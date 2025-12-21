import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time 

class Config:
    DATASET_DIR = 'H:/Учеба/Projects/Deep-learning/LR-1/archive/animals/animals'
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')
    IMG_SIZE = 128
    BATCH_SIZE = 128      
    VAL_SPLIT = 0.2   
    SEED = 42    
    # Параметры обучения
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_WORKERS = 4 # Количество потоков загрузки данных/как же я мучался с этим УЖАС работает только в .py
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset_metadata(base_dir):
    """
    Сканирует директорию и создает DataFrame с путями к изображениям и метками.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Директория не найдена: {base_dir}")   
    image_paths = []
    labels = []
    categories = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Обнаружено категорий: {len(categories)}")
    
    for category in tqdm(categories, desc="Индексация изображений"):
        cat_path = os.path.join(base_dir, category)  
        for img_filename in os.listdir(cat_path):
            if img_filename.lower().endswith(Config.VALID_EXTENSIONS):
                image_paths.append(os.path.join(cat_path, img_filename))
                labels.append(category)
                
    df = pd.DataFrame({
        'image_filepath': image_paths, 
        'label': labels
    }) 
    return df

def save_model_for_production(model, path="model.pth"):
    print(f"Экспорт модели в {path}")
    model.eval()
    device = next(model.parameters()).device  # Запоминаем устройство
    model.cpu()
    
    example_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    try:
        traced_model = torch.jit.trace(model, (example_input,))
        torch.jit.save(traced_model, path)
        print(f"Успешно! Модель сохранена.")
        model.to(device)  # Возвращаем модель обратно на устройство
        return True
    except Exception as e:
        print(f"Ошибка при экспорте модели: {e}")
        model.to(device)  # Возвращаем даже в случае ошибки
        return False
    
class EfficientAnimalDataset(Dataset):
    """
    Оптимизированный датасет. 
    Хранит пути и метки в списках, а не в DataFrame, чтобы избежать 
    проблем с сериализацией (зависаний) на Windows при multiprocessing.
    """
    def __init__(self, dataframe, transform=None):
        self.transform = transform
        
        # Конвертируем колонки в списки сразу при инициализации
        self.image_paths = dataframe['image_filepath'].tolist()
        self.labels = dataframe['label_idx'].tolist() # Берем числовые индексы
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx] 
        try:
            image = Image.open(img_path).convert('RGB') 
            if self.transform:
                image = self.transform(image)         
            return image, label     
        except Exception as e:
            print(f"Ошибка чтения файла {img_path}: {e}")
            return torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE)), label    

def train():

    print(f"Устройство для вычислений: {Config.DEVICE}")

    # Формирование датафрейма
    df = create_dataset_metadata(Config.DATASET_DIR)

    print("\nСводка по датасету:")
    print(f"Всего изображений: {len(df)}")
    print(f"Размерность данных: {df.shape}")
    print("-" * 30)
    print(df.head())
    print("-" * 30)

    # Создание маппинга классов
    classes = sorted(df['label'].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    num_classes = len(classes)

    # Добавляем числовой индекс класса сразу в DataFrame оптимизация)
    df['label_idx'] = df['label'].map(class_to_idx)

    print(f"Инициализация завершена. Количество классов: {num_classes}")

    # %% [markdown]
    # ### 2. ПОДГОТОВКА ДАННЫХ И DATALOADER

    # %%


    # Для обучения добавляем шум и искажения
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Немного меняем цвета
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Для валидации только ресайз и нормализация
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # Гарантирует, что редкие животные попадут и в train, и в val
    train_df, val_df = train_test_split(
        df, 
        test_size=Config.VAL_SPLIT, 
        stratify=df['label'], 
        random_state=Config.SEED
    )

    print(f"Разбиение данных: Train {len(train_df)} | Val {len(val_df)}")
    train_dataset = EfficientAnimalDataset(train_df, transform=train_transform)
    val_dataset = EfficientAnimalDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
        prefetch_factor=2 if Config.NUM_WORKERS > 0 else None  # Загружает данные заранее
    )
    print(Config.NUM_WORKERS)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
        prefetch_factor=2 if Config.NUM_WORKERS > 0 else None  # Загружает данные заранее
    )
    print("DataLoaders готовы к работе.")

    # %% [markdown]
    # ### 3. АРХИТЕКТУРА НЕЙРОСЕТИ И ОПТИМИЗАЦИЯ

    # %%
    class AnimalCNN(nn.Module):
        def __init__(self, num_classes):
            super(AnimalCNN, self).__init__()
            # Извлекаем простые признаки (грани, углы)
            # 128x128x3 -> 64x64x32
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2) 
            )
            # Текстуры и простые формы
            # 64x64 -> 32x32
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            # Сложные паттерны
            # 32x32 -> 16x16
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            # Высокоуровневые признаки
            # 16x16 -> 8x8
            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            # Классификатор
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(
                nn.Linear(256 * 8 * 8, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5), # Дропаем половину нейронов
                nn.Linear(1024, num_classes)
            )
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x
        
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # %% [markdown]
    # ### 4. ЦИКЛ ОБУЧЕНИЯ И ВАЛИДАЦИИ

    # %%
    print(f"Инициализация модели для {num_classes} классов")
    model = AnimalCNN(num_classes=num_classes).to(Config.DEVICE)

    total_params = count_parameters(model)
    print(f"Модель создана. Всего обучаемых параметров: {total_params:,}")

    # Лосс и Оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Добавим Scheduler опционально, но полезно для качества
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    print("Оптимизатор и функция потерь готовы.")
    print(f"Запуск обучения на {Config.NUM_EPOCHS} эпох")
    print(f"Устройство: {Config.DEVICE}")
    print("-" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'LR':^9} | {'GPU Mem':^9}")
    print("-" * 80)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    total_start = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False, unit="batch")
        
        for images, labels in pbar:
            images = images.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
        
        # Усредняем метрики за эпоху
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = correct / total
        
        # Валидация 
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        # Обновление Learning Rate
        if 'scheduler' in locals():
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = Config.LEARNING_RATE

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # Сохранение лучшей модели
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), "best_animal_model.pth")
            save_msg = f"-> Saved Best ({best_acc:.2%})"
        else:
            save_msg = ""
        print(next(model.parameters()).device)
        # Сколько памяти занято на GPU
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        epoch_time = time.time() - epoch_start

        print(f"{epoch+1:^7} | {avg_train_loss:^10.4f} | {avg_train_acc:^9.2%} | "
                f"{avg_val_loss:^10.4f} | {avg_val_acc:^9.2%} | {current_lr:^9.1e} | "
                f"{gpu_mem:^9.2f}GB {save_msg}")

    total_duration = (time.time() - total_start) / 60
    print("-" * 80)
    print(f"Обучение завершено. Время: {total_duration:.1f} мин. Лучшая точность: {best_acc:.2%}")

    # %% [markdown]
    # ### 5. АНАЛИЗ РЕЗУЛЬТАТОВ И ЭКСПОРТ

    # %%
    print("Построение графиков обучения")

    plt.figure(figsize=(15, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.title('Динамика Loss (Потери)')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Target (50%)') # Целевая отметка
    plt.title('Динамика Accuracy (Точность)')
    plt.xlabel('Эпохи')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Визуальная проверка
    def imshow(tensor, ax, title=None, color='black'):
        """Вспомогательная функция для отрисовки тензора"""
        img = tensor.cpu().clone()
        img = img * 0.5 + 0.5
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        if title:
            ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')

    dataiter = iter(val_loader)
    images, labels = next(dataiter)

    # Первые 5 примеров
    num_samples = 5
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Прогоняем через модель
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(Config.DEVICE))
        _, predicted = torch.max(outputs, 1)

    print(f"Примеры предсказаний на тестовых данных:")
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))

    for i in range(num_samples):
        true_idx = labels[i].item()
        pred_idx = predicted[i].item()
        
        true_label = idx_to_class[true_idx]
        pred_label = idx_to_class[pred_idx] 
        is_correct = true_idx == pred_idx
        color = 'green' if is_correct else 'red'
        title_text = f"True: {true_label}\nPred: {pred_label}"
        
        imshow(images[i], axes[i], title=title_text, color=color)

    plt.show()

    # Сохранение (Export)
    save_model_for_production(model)

if __name__ == '__main__':
    # Эта часть выполняется ТОЛЬКО в главном процессе
    
    # Исправление для Windows и Jupyter при использовании multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Запуск основной логики
    train()


