# Запуск
Чтобы воспроизвести наш результат, надо воспроизвести следующие шаги:
    <li> Убедиться, что у вас стоит python3.6 или выше </li>
    <li> Установить зависимости:
    
    pip install -r requirements.txt 
</li>
    <li> Расположить данные в правильной папке: скопировать файлы <b>train.csv</b> и <b>test.csv</b> в подпапку <b>data</b>. В итоге должна получиться следующая структура: 
    
    raif_gg_hack
        data
            train.csv
            test.csv
        raif_hack
            ...
        train.py
        predict.py
        validation_split.py
        ...
</li>
    <li> Выделить валидационный сет из обучающей выборки:

    python3 validation_split.py

</li>
    <li> Запустить обучение:

    python3 train.py --model_path model.pkl --val
</li>
    <li> Запустить инференс:
    
    python3 predict.py --model_path model.pkl --test_data data --output final_submission.csv
</li>
    <li> Предсказания содержатся в файле <b>final_submission.csv</b></li>
</ol>