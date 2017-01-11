#!/usr/bin/env bash
#скрипт запуска тестов luigi
[ -z $1 ] && echo 'Пропущен путь к тестируемому модулю' && exit 1
 ! [ -e $1 ]  && echo "$1 - модуль для тестов не найден!" && exit 1

CMD="$1"

TEST_DIR=./for_test
DATA_DIR=$TEST_DIR
RESULT_DIR=$TEST_DIR

echo ""
echo "Setup test workflow..."
echo "======================"
echo ""

! [ -e $TEST_DIR ] && mkdir $TEST_DIR

export DATA_DIR
export RESULT_DIR

python $CMD

#если третий аргумент ключь -b - то мы не удаляем временные файлы
echo ""
echo "======================"
if  ! [[ $2 == '-b' ]]; then
    echo "Delete all tests files..."
    rm -r $TEST_DIR
else 
    echo "Delete all tests files was denaded."
fi
echo ""
