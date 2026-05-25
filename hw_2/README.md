# Домашнее задание 2: Минимальный детектор на DETR

## Эксперимент

### Данные
**COCO 2017 subset**, 10 классов:
`person, bicycle, car, motorcycle, bus, truck, cat, dog, horse, bird`

- 2000 train / 500 val изображений
- category_id переиндексирован в плотный [0..9]

### Модель
- **DETR ResNet-50** (`facebook/detr-resnet-50`) — backbone ResNet-50 + transformer encoder-decoder
- 100 object queries
- Bipartite matching loss (Hungarian algorithm)
- Set prediction loss: classification CE + L1 + GIoU + auxiliary losses
- Classification head заменён под 10 классов через `ignore_mismatched_sizes=True`

## Гиперпараметры

| Параметр | Значение | Зачем |
|---|---|---|
| `BATCH_SIZE` | 4 | DETR требователен к памяти |
| `NUM_EPOCHS` | 10 | достаточно для fine-tuning |
| `LR` | 1e-4 | для transformer и heads |
| `LR_BACKBONE` | 1e-5 | в 10x меньше — не «ломаем» ImageNet-фичи |
| `WEIGHT_DECAY` | 1e-4 | AdamW |
| `CLIP_MAX_NORM` | 0.1 | gradient clipping, обязателен для DETR |
| Scheduler | CosineAnnealingLR | плавнее, чем step |
| Optimizer | AdamW | стандарт для трансформеров |
| `NUM_WORKERS` | 0 | классы из ячеек Jupyter не пиклятся в воркеры |

## Финальные метрики

| Метрика | Значение |
|---|---|
| **mAP @[0.5:0.95]** | **0.462** |
| **mAP50** | **0.693** |
| mAP75 | 0.458 |
| mAP small | 0.272 |
| mAP medium | 0.471 |
| mAP large | 0.641 |
| AR@100 | 0.617 |
| Best val loss | 0.948 (epoch 9) |

### Per-class AP

| Класс | AP | AP50 |
|---|---|---|
| dog | 0.803 | 0.943 |
| horse | 0.610 | 0.754 |
| bus | 0.543 | 0.673 |
| motorcycle | 0.508 | 0.789 |
| person | 0.504 | 0.790 |
| bird | 0.493 | 0.782 |
| truck | 0.352 | 0.567 |
| bicycle | 0.284 | 0.541 |
| car | 0.270 | 0.569 |
| cat | 0.256 | 0.525 |

### Распределение ошибок (всего 2395 FP, 435 FN, 1698 TP)

| Тип ошибки | Count | Доля FP |
|---|---|---|
| localization (тот же класс, IoU∈[0.1, 0.5)) | 963 | 40.2% |
| background (IoU<0.1) | 891 | 37.2% |
| duplicate (GT уже сматчен) | 437 | 18.2% |
| cls+loc | 54 | 2.3% |
| classification (объект найден, класс перепутан) | 50 | 2.1% |

**78% ошибок — это localization + background**, то есть проблема не в том, что модель видит, а в том, где и насколько уверенно.

### Главная путаница в классификации

Из 50 ошибок classification, 39 (78%) — это путаница внутри группы транспорта:
- **truck → car: 17 ошибок**
- car → truck: 11
- bus → truck: 8
- bus → car: 3


## Что логируется в TensorBoard

**По шагам (`train_step/`):** `loss_total`, `loss_ce`, `loss_bbox`, `loss_giou`, `lr`, `lr_backbone`

**По эпохам (`train_epoch/`, `val_epoch/`):** усреднённые те же компоненты

## Профилирование

Запускается до основного цикла: 2 wait + 2 warmup + 6 active шагов. Trace оборачивает 3 секции через `record_function`: `data_to_device`, `forward`, `backward`.

Trace сохраняется в `profiler/*.pt.trace.json` — формат kineto JSON.

## Error Analysis

Каждый FP классифицируется по правилам в духе TIDE:

| Тип ошибки | Условие |
|---|---|
| **TP** | IoU≥0.5 с GT того же класса (и GT ещё не сматчен) |
| **localization** | IoU∈[0.1, 0.5) с GT того же класса |
| **classification** | IoU≥0.5, но класс не совпадает |
| **cls+loc** | IoU∈[0.1, 0.5) и класс не совпадает |
| **duplicate** | IoU≥0.5 с GT нужного класса, но он уже занят |
| **background** | IoU<0.1 с любым GT |
| **FN** | GT, не сматченный ни с одним предсказанием |


## Ключевые наблюдения

1. **CE сходится быстрее, чем bbox/giou** — классификация при fine-tuning проще, чем регрессия координат
2. **Лёгкий оверфит** к 10-й эпохе по `loss_bbox` (train 0.029 vs val 0.043) — нужны аугментации для дальнейшего обучения
3. **Маленькие объекты (`mAP_small=0.272`)** — слабая зона DETR
4. **Транспорт (car/truck/bus)** — главный источник classification errors
5. **Много background-ошибок** на сценах с толпами — модель видит больше, чем размечено в GT
6. **78% ошибок — локализационные и background**, не классификационные
