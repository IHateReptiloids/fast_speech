# Домашнее задание по TTS
[Условие](https://github.com/markovka17/dla/tree/2021/hw3_tts)

## Установка
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone https://github.com/NVIDIA/waveglow.git
```

## Обзор репозитория
1. В папке `src/` находится почти весь исходный код.
2. В папке `data/` находится разбиение `LJSpeech`-а на `train` и `val`. Это разбиение сгененировано командой `python generate_trainval_split.py`
3. Скрипт `train.py` запускает обучение на `LJSpeech`. Все необходимые данные он подгружает, если их нет.
4. Скрипт `test.py` тестирует модель на 3-ех заранее выбранных предложениях. Он загружает веса модели автоматически. Его можно запускать командой `python test.py`.

## Выполненная работа
1. Написана модель `FastSpeech`, ее код находится в `src/models/fast_speech.py`.
2. Выполнено обучение модели. Все логи можно найти [здесь](https://wandb.ai/_username_/fast_speech). Отчет о проделанной работе можно найти [здесь](https://wandb.ai/_username_/fast_speech/reports/-TTS--VmlldzoxMzAwNDUw).
3. Модель протестирована на 3-ех предложениях из условия. Результаты можно увидеть здесь(тут ссылка).
