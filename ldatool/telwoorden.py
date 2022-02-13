#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-,

file = open('woordentextnrcdata.txt', 'r')
read_data = file.read()
per_word = read_data.split()

print('Total Words:', len(per_word))
