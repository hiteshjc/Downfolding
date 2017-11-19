#!/bin/sh
#This is to combine all the tex files into a single tex file. 
##
sed -e '/\input{intro.tex}/ {' -e 'r intro.tex' -e 'd' -e '}' \
    -e '/\input{theory.tex}/ {' -e 'r theory.tex' -e 'd' -e '}' \
    -e '/\input{threeband.tex}/ {' -e 'r threeband.tex' -e 'd' -e '}' \
    -e '/\input{hchain.tex}/ {' -e 'r hchain.tex' -e 'd' -e '}' \
    -e '/\input{graphene.tex}/ {' -e 'r graphene.tex' -e 'd' -e '}' \
    -e '/\input{fese.tex}/ {' -e 'r fese.tex' -e 'd' -e '}' \
    -e '/\input{conclusion.tex}/ {' -e 'r conclusion.tex' -e 'd' -e '}' \
    -e '/\input{appendix.tex}/ {' -e 'r appendix.tex' -e 'd' -e '}' \
    main.tex > manuscript.tex
