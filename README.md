# Yandex Cup ML Challenge '23
## RecSys Track

[**Задача**](https://yandex.ru/cup/ml/): Предсказание музыкального жанра по звучанию трека. Каждый трек описывается последовательностью ембедингов размерности 768, посчитанных по фрагменту аудио трека фиксированной длины.

**Public Avg PR**: *0.3067* \
**Private Avg PR**: *0.3114*

To reproduce the results:
1. `conda create -n ya python=3.10`, `source activate ya`, `pip install -r requirements.txt`
2. Run scripts: `scripts/run_train_models_0.sh`,`scripts/run_train_models_1.sh`,`scripts/run_train_models_2.sh`,`scripts/run_train_models_3.sh`
3. Average results using  `average.ipynb`

**Leaderboard:** 

<img width="618" alt="image" src="https://github.com/ArtemVazh/YandexCupML23/assets/44724467/ac98a2e3-730c-4019-8017-b389084574db">




