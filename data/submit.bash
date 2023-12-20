for kind in RNN RNNA RNNE GRU GRUA GRUE LSTM LSTMA LSTME
#for K in 5
do
	for t in 10 20 30 40 50
	do
		sbatch --export=ALL,kind=$kind,t=$t -a 0-4 newRunThis.sb 
	done
	sleep 1
done
