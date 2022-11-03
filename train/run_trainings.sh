python national_forecaster.py --batch 1 --num_workers 8 --fp16 --steps 8 --accumulate 16 --num_gpu 0 --hrv --sun &
python national_forecaster.py --batch 1 --num_workers 12 --fp16 --steps 8 --accumulate 16 --num_gpu 1 --hrv --sat --sun &
python national_forecaster.py --batch 1 --num_workers 16 --fp16 --steps 8 --accumulate 16 --num_gpu 2 --hrv --sat --nwp --sun &
python national_forecaster.py --batch 1 --num_workers 8 --fp16 --steps 8 --accumulate 16 --num_gpu 3 --nwp --sun &
python national_forecaster.py --batch 1 --num_workers 12 --fp16 --steps 8 --accumulate 16 --num_gpu 4 --nwp --sat --sun &
python national_forecaster.py --batch 1 --num_workers 8 --fp16 --steps 8 --accumulate 16 --num_gpu 5 --sat --sun

