# Determining the position of rail fasteners from video

Цель проекта: определение рельсовых креплений по видеосъемке.

Участники:
* Зенков Мирослав РИМ-120906
* Маркин Михаил РИМ-120907
* Соломеин Александр РИМ-120907

Принцип работы:
* Пользователь загружает видео mp4 через форму;
* Видео обрабатывается на бэке;
* Пользователь получает видео с распознанными креплениями.

## Методы API

Отправка видео. Принимается видео mp4 формата с соответствующим mimetype 
`@app.post("/upload")`

Пример ответа:
```  
{
  "hash": 8729932479735
}
``` 
Где hash - хэш по которому сохраняется файл

Проверка статуса обработки видео. file_hash - хэш полученный из ответа `/upload`

`@app.get('/status/{file_hash}')`

Пример ответа:
```  
{
  "status": "FINISH"
}
``` 

Загрузка видео. file_hash - хэш полученный из ответа `/upload`

`@app.get('/download/{file_hash}')`
В ответ высылается файл размеченного видео

## Запуск в Docker

`docker-compose build && docker-compose up`
