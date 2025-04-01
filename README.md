## Etape 1 : Make
`make`

## Etape 2 : Run l'exo
`mpirun -np <nb threads> ./lbm -e <exo n°>`

## Epate 3 : Créer le Gif
`./gen_animate_gif.sh`<br>
`./gen_animate_gif.sh <raw file> <gif name>`

## Tester un exercie : 
`./check_comm -e <exo n°>`

## Exemple for exe 1
`clear && make && mpirun -np 4 ./lbm -e 1 && ./gen_animate_gif.sh`
