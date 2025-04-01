## Etape 1 : Make
`make`

## Etape 2 : Run l'exo
`mpirun -np <nb threads> ./lbm -e <exo n°>`

## Epate 3 : Créer le Gif
`./gen_animate_gif.sh`
`./gen_animate_gif.sh <raw file> <gif name>`

## Exemple for exe 1
`clear && make && mpirun -np 4 ./lbm -e 1 && ./gen_animate_gif.sh`
