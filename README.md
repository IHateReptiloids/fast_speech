# Домашнее задание по TTS
[Условие](https://github.com/markovka17/dla/tree/2021/hw3_tts)

## Установка
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Выполненная работа
1. Написана модель `fastspeech`, ее код находится в `src/models/fast_speech.py`.
2. Написан скрипт `overfit.py`, который переобучает модель на одном батче. Переобучение также демонстрируется в ноутбуке `overfit.ipynb`. Веса получившейся модели, а также батч, на котором модель переобучалась, сохранены в `checkpoints/overfitted`.
