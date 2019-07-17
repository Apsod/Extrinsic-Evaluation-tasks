#!/usr/bin/env bash

trap "exit" INT
for vector_path in "$@"
do
  echo "VECTOR PATH: $vector_path"
  for method in snli subjective relation sentence_sentiment document_sentiment
  do
    echo "$method"
    exeval --log --vector_path $vector_path $method
  done

  for task in pos ner chunk
  do
    echo "sequence_labeling $task"
    exeval --log --vector_path $vector_path sequence_labeling --task $task
  done
done
