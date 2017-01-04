#!/usr/bin/env bash
#Скрипт для запуска manage.py в окурджении которе есть в контейнере
#   - первым аргументом получает путь к файлу с переменными окружения;
#   - вторым аргументом получает команду для manage.py

[ -z $1 ] && echo 'Пропущен путь к файлу с переменными окружения' && exit 1
[ -z $2 ] && echo 'Пропущен аргумент для luigi' && exit 1
 ! [ -e $1 ]  && echo "$1 - файл с переменными не найден!" && exit 1

CMD="$2 $3 "
ENV=`cat $1`

for env in ${ENV}; do
    export $env
done

python -m luigi --module $CMD --local-scheduler
